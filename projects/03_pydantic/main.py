from pydantic import BaseModel, field_validator, model_validator



class Student(BaseModel):
    name: str
    age: int
    subject: str
    cast: str
    city: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v < 10 or v > 30:
            raise ValueError("Age must be between 10 and 30")
        return v

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v):
        if v not in ["AI", "Math", "Science"]:
            raise ValueError("Subject must be AI, Math, or Science")
        return v

    @field_validator("cast")
    @classmethod
    def capitalize_cast(cls, v):
        return v.capitalize()

    @model_validator(mode="after")
    @classmethod
    def check_subject_age(cls, values):
        age = values.age
        subject = values.subject
        if subject == "AI" and age < 18:
            raise ValueError("AI subject is only allowed for age 18 and above")
        return values
