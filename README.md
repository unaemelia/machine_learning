# Machine Learning Price Prediction for RAM

This project demonstrates a simple machine learning application using the ML.NET library to predict the price of a computer based on its RAM size. The project sets up a basic linear regression model using sample data to estimate the price of a computer.

## Project Overview

The project utilizes the ML.NET library to perform the following tasks:

- Define a set of training data representing computer prices based on different RAM sizes.
- Create a machine learning pipeline for regression using the `Sdca` (Stochastic Dual Coordinate Ascent) trainer.
- Train the model using the provided data.
- Predict the price of a computer with a given amount of RAM.

## Code Breakdown

### Program.cs

The main code resides in the `Program.cs` file, where the entire machine learning pipeline is set up and executed.

- **MLContext Initialization**: Initializes the ML.NET environment.
  ```csharp
  MLContext mlContext = new MLContext();
  ```
  
  - Training Data: This array represents the training data with known prices for different RAM sizes.
  ```csharp
  ComputerData[] computerData = 
  {
      new ComputerData() {RAM=8, Price=800},
      new ComputerData() {RAM=16, Price=1000},
      new ComputerData() {RAM=32, Price=1500},
      new ComputerData() {RAM=64, Price=2500},
  };
  ```
  
  - Data Preperation: Loads the training data into an IDataView for processing.
  ```csharp
  IDataView trainingData = mlContext.Data.LoadFromEnumerable(computerData);
  ```

  - Pipeline Creation: Sets up the machine learning pipeline with a feature transformation and a regression trainer.
  ```csharp
  var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "RAM" })
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));
  ```

  - Model Training: Trains the model using the provided training data.
  ```csharp
  var model = pipeline.Fit(trainingData);
  ```

  - Price Prediction: Predicts the price for a computer with 100GB of RAM and outputs the result to the console.
  ```csharp
  ComputerData randomAccessMemory = new ComputerData() { RAM = 100 };
  Prediction price = mlContext.Model.CreatePredictionEngine<ComputerData, Prediction>(model).Predict(randomAccessMemory);
  Console.WriteLine($"Predicted price for RAM: {randomAccessMemory.RAM} price = {price.Price:C}");
  ```

- **Data Classes**:
  - ComputerData Class: Represents the input data structure with RAM and Price properties.
  ```csharp
  public class ComputerData
  {
    public float RAM { get; set; }
    public float Price { get; set; }
  }
  ```

  - Prediction Class: Holds the predicted price, with the Price property mapped to the model's output score.
  ```csharp
  public class Prediction
  {
    [ColumnName("Score")]
    public float Price { get; set; }
  }
  ```

## Running the Project
1. Clone the Repository:
```bash
git clone https://github.com/unaemelia/machine_learning.git
```

2. Navigate to the Project Directory:
```bash
cd machine_learning
```

3. Build and Run the Application: Ensure you have the .NET SDK installed. Run the following command to execute the program:
```bash
dotnet run
```

Expected Output: The program will output the predicted price for a computer with 100GB of RAM in the console.

## Future Improvements
- `Expand the Dataset`: Add more data points with different RAM sizes and prices.
- `Enhance the Model`: Experiment with different regression algorithms or feature engineering techniques.
- `User Interaction`: Add functionality for users to input their own RAM values for prediction.
  

