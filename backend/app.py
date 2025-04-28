from fastapi import FastAPI
from pydantic import BaseModel
from backend.utils import predict_top_3
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Formation Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class UserInput(BaseModel):
    Age: int
    Sexe: int
    Moyenne_Lycée: float
    Filiere: int
    Autres_Clubs: int
    Projets_Realisés: int
    Evaluation_Bureau: float
    Soft_Skills: float
    Score_Entretien: float
    Experience_Professionnelle: int
    Indice_Engagement: float
    Cellule: str

@app.get("/")
def welcome():
    return {"message": "Formation Recommendation API"}

@app.post("/predict")
def predict(user: UserInput):
    return {"recommendations": predict_top_3(user.dict())}
