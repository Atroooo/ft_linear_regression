class LinearRegression:
    """Linear regression model using gradient descent."""

    def __init__(self, lr=0.001, epochs=1000):
        """Initialise the model.

        Args:
            lr (float, optional): Learning rate of the model.
                Defaults to 0.001.
            epochs (int, optional): Number of steps.
                Defaults to 1000.
        """
        self.learning_rate = lr
        self.epochs = epochs
        self.w = 0  # theta1
        self.b = 0  # theta0
        self.iterations = 0
        self.global_loss = 0

    def predict(self, input):
        """Predict the output.

        Args:
            input (int, float): Input value.

        Returns:
            float: Predicted output.
        """
        return self.w * input + self.b

    def calculate_loss(self):
        """Calculate the loss. (Mean Squared Error)

        Returns:
            float: return the loss.
        """
        global_loss = 0
        for i in range(0, self.data_len):
            predictions = self.predict(self.X[i])
            loss_i = (predictions - float(self.y[i])) ** 2
            global_loss += loss_i
        return (1 / (2 * self.data_len)) * global_loss

    def gradient_descent(self):
        """Gradient descent algorithm to update the weights and bias."""
        dw = float(0)
        db = float(0)

        for i in range(0, self.data_len):
            predictions = self.predict(self.X[i])
            db += predictions - self.y[i]
            dw += (predictions - self.y[i]) * self.X[i]

        tw = self.w - (self.learning_rate *
                       ((1 / self.data_len) * dw))
        tb = self.b - (self.learning_rate *
                       ((1 / self.data_len) * db))
        self.w = tw
        self.b = tb

    def train(self, X, y, print_every=100):
        """Train the model.

        Args:
            X (list): Car mileage.
            y (list): Car price.
            print_every (int, optional): Print the loss every n epochs.
                Defaults to 100.
        """
        self.X = X
        self.y = y
        self.data_len = len(X)

        for epoch in range(self.epochs):
            loss = self.calculate_loss()

            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'loss: {loss:.3f}',
                      f'w: {self.w:.3f}',
                      f'b: {self.b:.3f}')

            self.gradient_descent()
            self.iterations += 1
        print("weight =", self.w, "b =", self.b)

    def get_params(self):
        """Return the weights and bias.

        Returns:
            float: weights and bias.
        """
        return self.w, self.b
