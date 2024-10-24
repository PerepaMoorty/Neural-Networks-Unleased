import numpy as np
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0


def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

class BatchNorm:
    def __init__(self, dim, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.moving_mean = np.zeros(dim)
        self.moving_var = np.ones(dim)
        
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0) + self.eps
            
            
            momentum = 0.9
            self.moving_mean = momentum * self.moving_mean + (1 - momentum) * mean
            self.moving_var = momentum * self.moving_var + (1 - momentum) * var
            
            x_norm = (x - mean) / np.sqrt(var)
            self.cache = (x, x_norm, mean, var)
        else:
            x_norm = (x - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
        
        out = self.gamma * x_norm + self.beta
        return out
    
    def backward(self, dout):
        x, x_norm, mean, var = self.cache
        N = dout.shape[0]
        
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        dmean = np.sum(dx_norm * -1/np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, reg_type='l2', reg_lambda=0.01):
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        
        
        self.layers = len(hidden_sizes) + 1
        self.weights = []
        self.biases = []
        self.batch_norms = []
        
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size) * np.sqrt(2.0/prev_size))
            self.biases.append(np.zeros(hidden_size))
            self.batch_norms.append(BatchNorm(hidden_size))
            prev_size = hidden_size
        
        
        self.weights.append(np.random.randn(prev_size, output_size) * np.sqrt(2.0/prev_size))
        self.biases.append(np.zeros(output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        self.activations = [X]
        self.z_values = []
        
        
        for i in range(self.layers - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            
            z = self.batch_norms[i].forward(z, training)
            
            a = self.relu(z)
            self.activations.append(a)
        
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output, batch_size):
        dW = []
        db = []
        
        
        delta = output - y
        dW.append(np.dot(self.activations[-2].T, delta) / batch_size)
        db.append(np.sum(delta, axis=0) / batch_size)
        
        
        for i in range(self.layers - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T)
            delta = self.batch_norms[i].backward(delta)
            delta = delta * self.relu_derivative(self.z_values[i])
            
            dW.insert(0, np.dot(self.activations[i].T, delta) / batch_size)
            db.insert(0, np.sum(delta, axis=0) / batch_size)
        
        
        if self.reg_type == 'l2':
            for i in range(len(dW)):
                dW[i] += self.reg_lambda * self.weights[i]
        elif self.reg_type == 'l1':
            for i in range(len(dW)):
                dW[i] += self.reg_lambda * np.sign(self.weights[i])
        
        return dW, db
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(np.maximum(y_pred[range(m), y_true.argmax(axis=1)], 1e-10))
        loss = np.sum(log_likelihood) / m
        
        if self.reg_type == 'l2':
            reg_loss = 0
            for w in self.weights:
                reg_loss += np.sum(w * w)
            loss += 0.5 * self.reg_lambda * reg_loss
        elif self.reg_type == 'l1':
            reg_loss = 0
            for w in self.weights:
                reg_loss += np.sum(np.abs(w))
            loss += self.reg_lambda * reg_loss
            
        return loss
    
    def train(self, X, y, batch_size=128, learning_rate=0.001, epochs=10):
        n_samples = X.shape[0]
        
        
        m_weights = [np.zeros_like(w) for w in self.weights]
        v_weights = [np.zeros_like(w) for w in self.weights]
        m_biases = [np.zeros_like(b) for b in self.biases]
        v_biases = [np.zeros_like(b) for b in self.biases]
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        t = 0
        
        for epoch in range(epochs):
            
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            
            for i in range(0, n_samples, batch_size):
                t += 1
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                
                output = self.forward(batch_X, training=True)
                
                
                dW, db = self.backward(batch_X, batch_y, output, batch_size)
                
                
                for j in range(len(self.weights)):
                    
                    m_weights[j] = beta1 * m_weights[j] + (1 - beta1) * dW[j]
                    v_weights[j] = beta2 * v_weights[j] + (1 - beta2) * (dW[j] * dW[j])
                    m_hat = m_weights[j] / (1 - beta1**t)
                    v_hat = v_weights[j] / (1 - beta2**t)
                    self.weights[j] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    
                    
                    m_biases[j] = beta1 * m_biases[j] + (1 - beta1) * db[j]
                    v_biases[j] = beta2 * v_biases[j] + (1 - beta2) * (db[j] * db[j])
                    m_hat = m_biases[j] / (1 - beta1**t)
                    v_hat = v_biases[j] / (1 - beta2**t)
                    self.biases[j] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            
            train_predictions = self.forward(X, training=False)
            train_loss = self.compute_loss(y, train_predictions)
            train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == np.argmax(y, axis=1))
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Loss: {train_loss:.4f}")
            print(f"Accuracy: {train_accuracy:.4f}")


model = NeuralNetwork(
    input_size=784,  
    hidden_sizes=[512, 256],  
    output_size=10,
    reg_type='l2',
    reg_lambda=0.0001  
)


model.train(
    X_train,
    y_train,
    batch_size=64,  
    learning_rate=0.0005,  
    epochs=25  
)