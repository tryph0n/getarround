"""Pydantic schemas for prediction API."""

from pydantic import BaseModel, Field


class CarFeatures(BaseModel):
    """Features for a single car rental price prediction.

    Attributes:
        model_key: Car brand/model (e.g., "Citroen", "Peugeot", "BMW").
        mileage: Car mileage in kilometers.
        engine_power: Engine power in horsepower.
        fuel: Fuel type ("diesel", "petrol", "hybrid_petrol", "electro").
        paint_color: Car color.
        car_type: Type of car ("sedan", "hatchback", "suv", "van", etc.).
        private_parking_available: Has private parking.
        has_gps: Has GPS.
        has_air_conditioning: Has air conditioning.
        automatic_car: Is automatic transmission.
        has_getaround_connect: Has Getaround Connect feature.
        has_speed_regulator: Has speed regulator/cruise control.
        winter_tires: Has winter tires.
    """

    model_key: str = Field(..., examples=["Citroen"])
    mileage: int = Field(..., ge=0, examples=[100000])
    engine_power: int = Field(..., gt=0, examples=[120])
    fuel: str = Field(..., examples=["diesel"])
    paint_color: str = Field(..., examples=["black"])
    car_type: str = Field(..., examples=["sedan"])
    private_parking_available: bool = Field(default=False)
    has_gps: bool = Field(default=False)
    has_air_conditioning: bool = Field(default=False)
    automatic_car: bool = Field(default=False)
    has_getaround_connect: bool = Field(default=False)
    has_speed_regulator: bool = Field(default=False)
    winter_tires: bool = Field(default=False)


class PredictionInput(BaseModel):
    """Input schema for /predict endpoint."""

    cars: list[CarFeatures] = Field(
        ...,
        description="List of cars to predict prices for",
        min_length=1,
        max_length=50,
    )


class PredictionOutput(BaseModel):
    """Output schema for /predict endpoint."""

    prediction: list[int] = Field(
        ...,
        description="List of predicted rental prices per day in EUR",
    )
