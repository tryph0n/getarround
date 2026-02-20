"""Factory Boy factories for test data generation."""

import factory


class CarFeaturesFactory(factory.Factory):
    """Factory for generating car features."""

    class Meta:
        model = dict

    model_key = factory.Faker(
        "random_element",
        elements=["Peugeot", "Renault", "Citroen", "BMW", "Audi", "Volkswagen"],
    )
    mileage = factory.Faker("random_int", min=10000, max=200000)
    engine_power = factory.Faker("random_int", min=60, max=300)
    fuel = factory.Faker(
        "random_element", elements=["diesel", "petrol", "hybrid_petrol"]
    )
    paint_color = factory.Faker(
        "random_element",
        elements=["black", "white", "grey", "blue", "red", "silver"],
    )
    car_type = factory.Faker(
        "random_element",
        elements=["sedan", "hatchback", "suv", "van", "estate"],
    )
    private_parking_available = factory.Faker("boolean")
    has_gps = factory.Faker("boolean")
    has_air_conditioning = factory.Faker("boolean")
    automatic_car = factory.Faker("boolean")
    has_getaround_connect = factory.Faker("boolean")
    has_speed_regulator = factory.Faker("boolean")
    winter_tires = factory.Faker("boolean")


class LuxuryCarFactory(CarFeaturesFactory):
    """Factory for luxury car features."""

    model_key = "BMW"
    mileage = factory.Faker("random_int", min=10000, max=50000)
    engine_power = factory.Faker("random_int", min=200, max=400)
    private_parking_available = True
    has_gps = True
    has_air_conditioning = True
    automatic_car = True
    has_getaround_connect = True
    has_speed_regulator = True
    winter_tires = True


class BudgetCarFactory(CarFeaturesFactory):
    """Factory for budget car features."""

    model_key = "Renault"
    mileage = factory.Faker("random_int", min=100000, max=200000)
    engine_power = factory.Faker("random_int", min=60, max=100)
    private_parking_available = False
    has_gps = False
    has_air_conditioning = False
    automatic_car = False
    has_getaround_connect = False
    has_speed_regulator = False
    winter_tires = False
