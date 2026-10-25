import os
import glob
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt

import numpy as np
from PIL import Image
import io

from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

# --- Settings & Config ---
SECRET_KEY = "supersecretkey_for_jwt" # In production, use a secure env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URL = "sqlite:///" + os.path.join(BASE_DIR, "paddy_vision.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Paddy Vision API", description="AI-Based Nutrient Deficiency and Disease Prediction API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB Models ---
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)

class ScanHistoryDB(Base):
    __tablename__ = "scan_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    predicted_class = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Pydantic Models ---
class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str

class UserResponse(BaseModel):
    username: str
    full_name: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ScanResult(BaseModel):
    predicted_class: str
    confidence: float
    recommendation: dict

class ScanHistoryResponse(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    timestamp: datetime

# --- Load ML Model ---
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
model_files = glob.glob(os.path.join(MODEL_SAVE_DIR, "*.h5"))
model = None
if model_files:
    MODEL_PATH = max(model_files, key=os.path.getmtime)
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    print("Warning: No model found! Please ensure you have trained the model.")

class_labels = ['blast', 'healthy', 'insect', 'leaf_folder', 'tungro']

recommendations_db = {
    "blast": {
        "status": "Disease",
        "color": "Red",
        "fertilizers": ["Tricyclazole 75% WP", "Isoprothiolane 40% EC"],
        "preventive": "Avoid excessive nitrogen fertilizer. Keep field free of weeds.",
        "instructions": ["Mix 1g of Tricyclazole per liter of water.", "Spray uniformly on leaves.", "Repeat after 10 days if symptoms persist."]
    },
    "healthy": {
        "status": "Healthy",
        "color": "Green",
        "fertilizers": ["Standard NPK formulation as per schedule"],
        "preventive": "Continue current practices. Monitor regularly.",
        "instructions": ["Maintain proper water levels.", "Ensure balanced nutrient application."]
    },
    "insect": {
        "status": "Disease",
        "color": "Red",
        "fertilizers": ["Chlorpyrifos 20% EC", "Neem Oil"],
        "preventive": "Use light traps. Practice crop rotation.",
        "instructions": ["Spray Neem oil (5ml/liter) in the evening.", "For severe cases, use chemical insecticides as recommended by experts."]
    },
    "leaf_folder": {
        "status": "Disease",
        "color": "Red",
        "fertilizers": ["Cartap hydrochloride 50% SP", "Chlorantraniliprole 18.5% SC"],
        "preventive": "Avoid shade in the field. Remove grassy weeds.",
        "instructions": ["Apply Cartap at 1kg/acre.", "Ensure the spray reaches the folded leaves."]
    },
    "tungro": {
        "status": "Disease",
        "color": "Red",
        "fertilizers": ["Imidacloprid 17.8% SL (to control vectors)"],
        "preventive": "Plant resistant varieties. Destroy stubbles after harvest.",
        "instructions": ["Control green leafhoppers immediately.", "Uproot and burn infected plants to prevent spread."]
    }
}

# --- Dependencies ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# --- Auth Routes ---
@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = UserDB(username=user.username, hashed_password=hashed_password, full_name=user.full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: UserDB = Depends(get_current_user)):
    return current_user

# --- Prediction Route ---
@app.post("/predict", response_model=ScanResult)
async def predict(file: UploadFile = File(...), current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        contents = await file.read()
        uploaded_image = Image.open(io.BytesIO(contents)).convert("RGB")
        uploaded_image = uploaded_image.resize((299, 299))
        
        image_array = keras_image.img_to_array(uploaded_image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        
        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_class = class_labels[predicted_class_idx]
        
        # Save to history
        history_entry = ScanHistoryDB(
            user_id=current_user.id,
            predicted_class=predicted_class,
            confidence=confidence
        )
        db.add(history_entry)
        db.commit()
        
        return ScanResult(
            predicted_class=predicted_class,
            confidence=confidence,
            recommendation=recommendations_db.get(predicted_class, {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/history", response_model=List[ScanHistoryResponse])
def get_history(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(ScanHistoryDB).filter(ScanHistoryDB.user_id == current_user.id).order_by(ScanHistoryDB.timestamp.desc()).all()
