class Logistic():
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def __str__(self,x,y):
        
        return f"Logistic Regression Model:\nLearning Rate: {self.learning_rate}, Number of Iterations: {self.num_iterations}" 
    


    def predict():