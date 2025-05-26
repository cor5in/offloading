# src/algorithms/traffic_prediction.py
"""
BLSTM Traffic Prediction Implementation
Based on the paper's traffic prediction methodology
"""

import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.optimizers import RMSprop
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Traffic prediction will use simple patterns.")

class TrafficPredictionBLSTM:
    """Bidirectional LSTM for user traffic prediction"""
    
    def __init__(self, sequence_length=24, hidden_units=500, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.is_trained = False
        
        if TF_AVAILABLE:
            self.build_model()
        else:
            print("Using fallback traffic prediction (sine wave patterns)")
        
    def build_model(self):
        """Build BLSTM model architecture from paper"""
        if not TF_AVAILABLE:
            return
            
        self.model = Sequential([
            # First BLSTM layer with 500 hidden units
            Bidirectional(
                LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate),
                input_shape=(self.sequence_length, 1)
            ),
            
            # Second BLSTM layer with 500 hidden units  
            Bidirectional(
                LSTM(self.hidden_units, return_sequences=False, dropout=self.dropout_rate)
            ),
            
            # Fully connected layer
            Dense(self.hidden_units, activation='relu'),
            Dropout(self.dropout_rate),
            
            # Output layer for next day prediction (24 hours)
            Dense(24, activation='linear')
        ])
        
        # Use RMSprop optimizer as specified in paper
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
    def prepare_sequences(self, data):
        """Prepare sequences for training"""
        if len(data) < self.sequence_length + 24:
            raise ValueError(f"Need at least {self.sequence_length + 24} data points")
            
        X, y = [], []
        
        # Create overlapping sequences
        for i in range(len(data) - self.sequence_length - 23):
            # Input: previous sequence_length hours
            X.append(data[i:i+self.sequence_length])
            # Output: next 24 hours
            y.append(data[i+self.sequence_length:i+self.sequence_length+24])
            
        return np.array(X), np.array(y)
        
    def train(self, traffic_data, validation_split=0.2, epochs=50, batch_size=32, verbose=1):
        """Train BLSTM on historical traffic data"""
        if not TF_AVAILABLE or self.model is None:
            print("TensorFlow not available or model not built")
            return None
            
        try:
            # Prepare training data
            X, y = self.prepare_sequences(traffic_data)
            
            # Reshape for LSTM input (samples, timesteps, features)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Train the model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                shuffle=True
            )
            
            self.is_trained = True
            return history
            
        except Exception as e:
            print(f"Training failed: {e}")
            return None
            
    def predict_traffic(self, historical_sequence):
        """Predict next 24 hours of traffic"""
        if not TF_AVAILABLE or self.model is None or not self.is_trained:
            # Fallback to simple pattern-based prediction
            return self._fallback_prediction(historical_sequence)
            
        try:
            # Prepare input sequence
            if len(historical_sequence) < self.sequence_length:
                # Pad with zeros if insufficient history
                padded = np.zeros(self.sequence_length)
                padded[-len(historical_sequence):] = historical_sequence
                historical_sequence = padded
                
            # Take last sequence_length points
            sequence = np.array(historical_sequence[-self.sequence_length:]).reshape(1, self.sequence_length, 1)
            
            # Predict next 24 hours
            prediction = self.model.predict(sequence, verbose=0)
            return prediction[0]
            
        except Exception as e:
            print(f"Prediction failed: {e}, using fallback")
            return self._fallback_prediction(historical_sequence)
            
    def _fallback_prediction(self, historical_sequence):
        """Simple pattern-based prediction when TensorFlow is not available"""
        if len(historical_sequence) == 0:
            # Default daily pattern
            hours = np.arange(24)
            # Simple sine wave with peaks at 10 AM and 8 PM
            pattern = 0.5 + 0.3 * np.sin((hours - 6) * np.pi / 12) + 0.2 * np.sin((hours - 20) * np.pi / 4)
            return np.maximum(pattern, 0.1)  # Minimum traffic
            
        # Use recent average with daily pattern
        recent_avg = np.mean(historical_sequence[-24:]) if len(historical_sequence) >= 24 else np.mean(historical_sequence)
        
        hours = np.arange(24)
        # Daily pattern based on typical user behavior
        daily_pattern = np.array([
            0.2, 0.15, 0.1, 0.1, 0.15, 0.3,   # 0-5 AM: very low
            0.5, 0.7, 0.9, 1.0, 0.95, 0.9,   # 6-11 AM: morning rise
            0.85, 0.8, 0.75, 0.8, 0.85, 0.9,  # 12-5 PM: daytime
            1.2, 1.5, 1.3, 1.0, 0.7, 0.4      # 6-11 PM: evening peak
        ])
        
        # Scale by recent average and add some noise
        prediction = recent_avg * daily_pattern
        noise = 0.05 * np.random.normal(size=24)
        prediction += noise * prediction  # Proportional noise
        
        return np.maximum(prediction, 0.1)  # Minimum traffic
        
    def evaluate_prediction(self, test_data):
        """Evaluate prediction accuracy using MAE and RMSE"""
        if len(test_data) < self.sequence_length + 24:
            return {'mae': float('inf'), 'rmse': float('inf')}
            
        predictions = []
        actuals = []
        
        # Generate predictions for test data
        for i in range(len(test_data) - self.sequence_length - 23):
            input_seq = test_data[i:i+self.sequence_length]
            actual_next_day = test_data[i+self.sequence_length:i+self.sequence_length+24]
            predicted_next_day = self.predict_traffic(input_seq)
            
            predictions.extend(predicted_next_day)
            actuals.extend(actual_next_day)
            
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return {'mae': mae, 'rmse': rmse}
        
    def save_model(self, filepath):
        """Save trained model"""
        if TF_AVAILABLE and self.model is not None and self.is_trained:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save")
            
    def load_model(self, filepath):
        """Load pre-trained model"""
        if TF_AVAILABLE:
            try:
                self.model = tf.keras.models.load_model(filepath)
                self.is_trained = True
                print(f"Model loaded from {filepath}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("TensorFlow not available, cannot load model")

class SimpleTrafficPredictor:
    """Simple traffic predictor for when BLSTM is not available"""
    
    def __init__(self):
        self.user_patterns = {}
        
    def learn_pattern(self, user_id, traffic_history):
        """Learn traffic pattern for a specific user"""
        if len(traffic_history) >= 24:
            # Extract daily pattern from history
            daily_patterns = []
            for i in range(0, len(traffic_history) - 23, 24):
                daily_patterns.append(traffic_history[i:i+24])
                
            # Average daily pattern
            avg_pattern = np.mean(daily_patterns, axis=0)
            self.user_patterns[user_id] = avg_pattern
            
    def predict_user_traffic(self, user_id, hours_ahead=24):
        """Predict traffic for specific user"""
        if user_id in self.user_patterns:
            pattern = self.user_patterns[user_id]
            # Repeat pattern for requested hours
            full_prediction = np.tile(pattern, (hours_ahead // 24) + 1)
            return full_prediction[:hours_ahead]
        else:
            # Default pattern if user not learned
            return self._default_pattern(hours_ahead)
            
    def _default_pattern(self, hours):
        """Default traffic pattern"""
        daily_pattern = np.array([
            0.2, 0.15, 0.1, 0.1, 0.15, 0.3,   # Night
            0.5, 0.7, 0.9, 1.0, 0.95, 0.9,   # Morning
            0.85, 0.8, 0.75, 0.8, 0.85, 0.9,  # Afternoon  
            1.2, 1.5, 1.3, 1.0, 0.7, 0.4      # Evening
        ])
        
        full_pattern = np.tile(daily_pattern, (hours // 24) + 1)
        return full_pattern[:hours]

# Utility functions for traffic prediction
def generate_synthetic_traffic_data(days=30, users=10):
    """Generate synthetic traffic data for testing"""
    np.random.seed(42)  # For reproducibility
    
    traffic_data = {}
    hours_total = days * 24
    
    for user_id in range(users):
        user_data = []
        base_traffic = np.random.uniform(0.5, 2.0)
        
        for hour in range(hours_total):
            daily_hour = hour % 24
            
            # Create realistic daily pattern
            if 6 <= daily_hour <= 22:  # Active hours
                if 19 <= daily_hour <= 22:  # Evening peak
                    multiplier = 1.5 + 0.8 * np.sin((daily_hour - 19) * np.pi / 3)
                elif 7 <= daily_hour <= 9:  # Morning peak  
                    multiplier = 1.2 + 0.3 * np.sin((daily_hour - 7) * np.pi / 2)
                else:
                    multiplier = 0.8 + 0.2 * np.sin((daily_hour - 6) * np.pi / 16)
            else:  # Night hours
                multiplier = 0.1 + 0.1 * np.random.random()
                
            # Add noise
            noise = 0.1 * np.random.normal()
            traffic = max(0.1, base_traffic * (multiplier + noise))
            user_data.append(traffic)
            
        traffic_data[user_id] = user_data
        
    return traffic_data

if __name__ == "__main__":
    # Test traffic prediction
    print("Testing Traffic Prediction...")
    
    # Generate test data
    traffic_data = generate_synthetic_traffic_data(days=7, users=5)
    
    # Test BLSTM predictor
    predictor = TrafficPredictionBLSTM(sequence_length=24, hidden_units=50)  # Smaller for testing
    
    user_0_data = traffic_data[0]
    print(f"User 0 traffic data length: {len(user_0_data)}")
    
    if TF_AVAILABLE:
        # Train on first 5 days, test on last 2 days
        train_data = user_0_data[:120]  # 5 days
        test_data = user_0_data[120:]   # 2 days
        
        print("Training BLSTM...")
        history = predictor.train(train_data, epochs=10, verbose=0)
        
        if history:
            print("Evaluating prediction...")
            metrics = predictor.evaluate_prediction(user_0_data)
            print(f"MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}")
            
    # Test prediction
    recent_history = user_0_data[-48:]  # Last 2 days
    prediction = predictor.predict_traffic(recent_history)
    print(f"Predicted next 24 hours: {prediction[:5]}... (showing first 5 values)")