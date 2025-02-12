# app/models/resume_models.py

from typing import List, Optional
from pydantic import BaseModel

class PersonalInformation(BaseModel):
    full_name: str
    email: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    phone: Optional[str] = None
    address_string: Optional[str] = None
    country_of_residence: Optional[str] = None
    city_of_residence: Optional[str] = None
    zip: Optional[str] = None
    state_of_residence: Optional[str] = None
    age: Optional[str] = None
    is_us_citizen: bool = False

class Education(BaseModel):
    primary: Optional[str] = None
    secondary: Optional[str] = None
    graduation: Optional[str] = None
    post_graduation: Optional[str] = None
    phd: Optional[str] = None

class Location(BaseModel):
    country: Optional[str] = None
    short_code: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    zip: Optional[str] = None

class Certification(BaseModel):
    name_of_certification: str
    institution: str

class WorkExperience(BaseModel):
    job_role_name: str
    organization_name: str
    is_present: bool
    description: str
    start_month_year: str
    end_month_year: str

class GenerateData(BaseModel):
    is_remote: bool
    total_work_experience: str
    keywords: List[str]
    technical_skills: List[str]
    general_skills: List[str]
    licenses: List[str]
    industries: List[str]
    possible_job_roles: List[str]
    summary: str
    search_location: Optional[str] = None
    resume_text:str

class ResumeAnalysisResponse(BaseModel):
    personal_information: PersonalInformation
    education: Education
    location: Location
    certifications: List[Certification]
    work_experience: List[WorkExperience]
    generateData: GenerateData
