
Notes of Housing Predictor
========================

The model file was too big, run the `Housing.py` file, to create a model locally

About this file:
----------------------

    3. housingMedianAge: Median age of a house within a block; a lower number is a newer building

    4. totalRooms: Total number of rooms within a block

    5. totalBedrooms: Total number of bedrooms within a block

    6. population: Total number of people residing within a block

    7. households: Total number of households, a group of people residing within a home unit, for a block

    8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)

    9. medianHouseValue: Median house value for households within a block (measured in US Dollars)

    10. oceanProximity: Location of the house w.r.t ocean/sea


Ocean Proximity:
---------------------------
    NEAR BAY    = 0
    NEAR OCEAN  =  1
    <1H OCEAN   = 2
    INLAND      = 3
    ISLAND      = 4
    

Error Scores:
---------------------

(Mean + Standared Deviation)

    x LinearRegression  : 73980.63680247005
    
    x DecisionTreeRegressor  : 88266.61937420537
    
    ->RandomForestRegressor : 64114.286938955425  
            
 Final Error:
 -----------------
 (Mean + Standared Deviation)
         RFR : 63652.44494071868 
    
    
