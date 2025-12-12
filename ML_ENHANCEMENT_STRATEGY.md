# Technic ML Enhancement Strategy: Achieving World-Class Accuracy

## Executive Summary

To make Technic the **most accurate investment recommendation system on the market**, we need to enhance the already-strong ML infrastructure with cutting-edge techniques, rigorous validation, and continuous improvement loops. This document outlines a comprehensive strategy to achieve >60% win rate and >0.15 Information Coefficient (IC) - metrics that would place Technic in the top 1% of quantitative systems.

---

## Current ML Infrastructure Assessment

### ✅ **Strengths**
1. **Solid Foundation**
   - LightGBM and XGBoost models with proper train/val/test splits
   - Model registry with versioning and gating
   - Multi-horizon predictions (5d and 10d)
   - Regime and sector-specific models
   - Rolling window training for temporal robustness

2. **Good Practices**
   - Walk-forward validation (no lookahead bias)
   - Baseline comparison before promotion
   - Feature engineering (technical + fundamental)
   - Cross-sectional normalization

3. **Advanced Features**
   - Alpha blending (factor + ML)
   - Meta-models for win probability
   - SHAP explainability
   - Ensemble methods

### ⚠️ **Areas for Improvement**
1. **Feature Engineering**: Can be significantly enhanced
2. **Model Architecture**: Opportunity for deep learning
3. **Ensemble Methods**: Underutilized
4. **Alternative Data**: Limited integration
5. **Online Learning**: Not implemented
6. **Hyperparameter Optimization**: Manual tuning
7. **Model Monitoring**: Basic metrics only
8. **Backtesting**: Limited historical validation

---

## Enhancement Strategy: 10 Pillars of World-Class ML

### **Pillar 1: Advanced Feature Engineering**

#### 1.1 Technical Features (Expand from ~20 to 100+)
**Current**: Basic momentum, volatility, volume
**Enhanced**:

```python
# technic_v4/engine/feature_engine_v2.py

class AdvancedFeatureEngine:
    """
    World-class feature engineering for alpha prediction.
    Target: 100+ features across 8 categories.
    """
    
    def compute_all_features(self, df: pd.DataFrame, fundamentals: dict) -> pd.Series:
        features = {}
        
        # 1. Multi-timeframe momentum (15 features)
        features.update(self._momentum_features(df))
        
        # 2. Volatility regime (12 features)
        features.update(self._volatility_features(df))
        
        # 3. Volume patterns (10 features)
        features.update(self._volume_features(df))
        
        # 4. Price patterns (15 features)
        features.update(self._price_pattern_features(df))
        
        # 5. Microstructure (8 features)
        features.update(self._microstructure_features(df))
        
        # 6. Fundamental factors (20 features)
        features.update(self._fundamental_features(fundamentals))
        
        # 7. Alternative data (10 features)
        features.update(self._alternative_data_features(df))
        
        # 8. Cross-sectional (10 features)
        features.update(self._cross_sectional_features(df))
        
        return pd.Series(features)
    
    def _momentum_features(self, df: pd.DataFrame) -> dict:
        """
        Multi-timeframe momentum with acceleration and regime awareness.
        """
        close = df['Close']
        features = {}
        
        # Standard momentum (multiple horizons)
        for period in [5, 10, 21, 42, 63, 126, 252]:
            ret = close.pct_change(period).iloc[-1]
            features[f'mom_{period}'] = ret
            
            # Momentum acceleration (2nd derivative)
            if period >= 21:
                mom_prev = close.pct_change(period).iloc[-period]
                features[f'mom_accel_{period}'] = ret - mom_prev
        
        # Momentum consistency (% of positive days)
        for period in [21, 63]:
            returns = close.pct_change().tail(period)
            features[f'mom_consistency_{period}'] = (returns > 0).mean()
        
        # Momentum vs moving average
        for ma_period in [20, 50, 200]:
            ma = close.rolling(ma_period).mean().iloc[-1]
            features[f'price_vs_ma{ma_period}'] = (close.iloc[-1] / ma) - 1
        
        # Momentum rank (percentile over lookback)
        for period in [63, 126]:
            rolling_rets = close.pct_change(21).rolling(period)
            current_ret = close.pct_change(21).iloc[-1]
            features[f'mom_rank_{period}'] = (rolling_rets < current_ret).mean().iloc[-1]
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> dict:
        """
        Volatility regime, realized vs implied, skew, term structure.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        features = {}
        
        # Realized volatility (multiple horizons)
        for period in [5, 10, 20, 60]:
            returns = close.pct_change()
            vol = returns.tail(period).std() * np.sqrt(252)
            features[f'vol_realized_{period}'] = vol
        
        # Parkinson volatility (high-low range)
        for period in [20, 60]:
            hl_ratio = np.log(high / low)
            park_vol = hl_ratio.tail(period).std() * np.sqrt(252 / (4 * np.log(2)))
            features[f'vol_parkinson_{period}'] = park_vol
        
        # Volatility of volatility
        vol_20 = close.pct_change().rolling(20).std()
        features['vol_of_vol'] = vol_20.tail(60).std()
        
        # Volatility regime (current vs historical)
        vol_current = close.pct_change().tail(20).std() * np.sqrt(252)
        vol_60d = close.pct_change().tail(60).std() * np.sqrt(252)
        vol_252d = close.pct_change().tail(252).std() * np.sqrt(252)
        features['vol_regime_20_60'] = vol_current / vol_60d if vol_60d > 0 else 1.0
        features['vol_regime_20_252'] = vol_current / vol_252d if vol_252d > 0 else 1.0
        
        # Downside volatility (semi-deviation)
        returns = close.pct_change().tail(60)
        downside_returns = returns[returns < 0]
        features['vol_downside'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Upside volatility
        upside_returns = returns[returns > 0]
        features['vol_upside'] = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
        
        # Volatility skew (upside vs downside)
        if features['vol_downside'] > 0:
            features['vol_skew'] = features['vol_upside'] / features['vol_downside']
        else:
            features['vol_skew'] = 1.0
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> dict:
        """
        Volume patterns, accumulation/distribution, money flow.
        """
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        features = {}
        
        # Volume trends
        for period in [5, 20, 60]:
            vol_ma = volume.rolling(period).mean()
            features[f'volume_vs_ma{period}'] = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0
        
        # Dollar volume
        dollar_vol = (close * volume).rolling(20).mean().iloc[-1]
        features['dollar_volume_20d'] = dollar_vol
        
        # Volume momentum
        vol_20d_ago = volume.rolling(20).mean().iloc[-20] if len(volume) >= 40 else volume.iloc[0]
        vol_current = volume.rolling(20).mean().iloc[-1]
        features['volume_momentum'] = (vol_current / vol_20d_ago) - 1 if vol_20d_ago > 0 else 0
        
        # On-balance volume (OBV)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_ma = obv.rolling(20).mean()
        features['obv_vs_ma20'] = (obv.iloc[-1] / obv_ma.iloc[-1]) - 1 if obv_ma.iloc[-1] != 0 else 0
        
        # Accumulation/Distribution Line
        mfm = ((close - low) - (high - close)) / (high - low)  # Money Flow Multiplier
        mfm = mfm.fillna(0)
        ad_line = (mfm * volume).cumsum()
        ad_ma = ad_line.rolling(20).mean()
        features['ad_line_vs_ma20'] = (ad_line.iloc[-1] / ad_ma.iloc[-1]) - 1 if ad_ma.iloc[-1] != 0 else 0
        
        # Chaikin Money Flow
        for period in [20, 60]:
            mfv = mfm * volume
            cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
            features[f'cmf_{period}'] = cmf.iloc[-1]
        
        return features
    
    def _price_pattern_features(self, df: pd.DataFrame) -> dict:
        """
        Chart patterns, support/resistance, breakouts.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        features = {}
        
        # Distance from highs/lows
        for period in [20, 52, 252]:
            if len(close) >= period:
                period_high = high.tail(period).max()
                period_low = low.tail(period).min()
                features[f'dist_from_high_{period}'] = (close.iloc[-1] / period_high) - 1
                features[f'dist_from_low_{period}'] = (close.iloc[-1] / period_low) - 1
        
        # Breakout detection
        high_20 = high.rolling(20).max()
        features['breakout_20d'] = 1.0 if close.iloc[-1] >= high_20.iloc[-2] else 0.0
        
        high_60 = high.rolling(60).max()
        features['breakout_60d'] = 1.0 if close.iloc[-1] >= high_60.iloc[-2] else 0.0
        
        # Support/resistance strength
        # Count how many times price touched 20d high in last 60 days
        touches = (high.tail(60) >= high_20.tail(60) * 0.99).sum()
        features['resistance_strength'] = touches / 60
        
        # Bollinger Band position
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        upper_band = ma_20 + (2 * std_20)
        lower_band = ma_20 - (2 * std_20)
        band_width = upper_band - lower_band
        features['bb_position'] = (close.iloc[-1] - lower_band.iloc[-1]) / band_width.iloc[-1] if band_width.iloc[-1] > 0 else 0.5
        features['bb_width'] = band_width.iloc[-1] / ma_20.iloc[-1] if ma_20.iloc[-1] > 0 else 0
        
        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['rsi_14'] = rsi.iloc[-1]
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd.iloc[-1]
        features['macd_signal'] = signal.iloc[-1]
        features['macd_histogram'] = (macd - signal).iloc[-1]
        
        return features
    
    def _microstructure_features(self, df: pd.DataFrame) -> dict:
        """
        Intraday patterns, bid-ask spread proxies, price impact.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']
        features = {}
        
        # Intraday range
        for period in [5, 20]:
            intraday_range = ((high - low) / close).rolling(period).mean()
            features[f'intraday_range_{period}'] = intraday_range.iloc[-1]
        
        # Gap analysis
        gap = (open_price - close.shift(1)) / close.shift(1)
        features['gap_today'] = gap.iloc[-1]
        features['gap_avg_20d'] = gap.tail(20).mean()
        
        # Close vs open (intraday momentum)
        intraday_ret = (close - open_price) / open_price
        features['intraday_momentum'] = intraday_ret.iloc[-1]
        features['intraday_momentum_avg_20d'] = intraday_ret.tail(20).mean()
        
        # High-low spread (liquidity proxy)
        hl_spread = (high - low) / close
        features['hl_spread_20d'] = hl_spread.tail(20).mean()
        
        # Price impact (range vs volume)
        price_impact = hl_spread / np.log1p(df['Volume'])
        features['price_impact_20d'] = price_impact.tail(20).mean()
        
        return features
    
    def _fundamental_features(self, fundamentals: dict) -> dict:
        """
        Value, quality, growth, leverage factors.
        """
        features = {}
        
        if not fundamentals:
            # Return zeros if no fundamentals available
            return {f'fund_{i}': 0.0 for i in range(20)}
        
        # Value factors
        features['pe_ratio'] = fundamentals.get('pe_ratio', np.nan)
        features['pb_ratio'] = fundamentals.get('pb_ratio', np.nan)
        features['ps_ratio'] = fundamentals.get('ps_ratio', np.nan)
        features['pcf_ratio'] = fundamentals.get('pcf_ratio', np.nan)
        features['ev_ebitda'] = fundamentals.get('ev_ebitda', np.nan)
        features['ev_sales'] = fundamentals.get('ev_sales', np.nan)
        features['earnings_yield'] = 1 / features['pe_ratio'] if features['pe_ratio'] > 0 else 0
        features['fcf_yield'] = fundamentals.get('fcf_yield', np.nan)
        
        # Quality factors
        features['roe'] = fundamentals.get('roe', np.nan)
        features['roa'] = fundamentals.get('roa', np.nan)
        features['roic'] = fundamentals.get('roic', np.nan)
        features['gross_margin'] = fundamentals.get('gross_margin', np.nan)
        features['operating_margin'] = fundamentals.get('operating_margin', np.nan)
        features['net_margin'] = fundamentals.get('net_margin', np.nan)
        
        # Growth factors
        features['revenue_growth'] = fundamentals.get('revenue_growth', np.nan)
        features['earnings_growth'] = fundamentals.get('earnings_growth', np.nan)
        features['fcf_growth'] = fundamentals.get('fcf_growth', np.nan)
        
        # Leverage factors
        features['debt_to_equity'] = fundamentals.get('debt_to_equity', np.nan)
        features['current_ratio'] = fundamentals.get('current_ratio', np.nan)
        features['interest_coverage'] = fundamentals.get('interest_coverage', np.nan)
        
        return features
    
    def _alternative_data_features(self, df: pd.DataFrame) -> dict:
        """
        Sentiment, news, insider activity, institutional ownership.
        """
        features = {}
        
        # Placeholder for alternative data integration
        # These would be populated from external sources
        features['news_sentiment_7d'] = 0.0  # -1 to 1
        features['news_volume_7d'] = 0.0  # Count of articles
        features['social_sentiment_7d'] = 0.0  # Twitter/Reddit sentiment
        features['insider_buying_90d'] = 0.0  # Net insider purchases
        features['institutional_ownership_change'] = 0.0  # QoQ change
        features['short_interest'] = 0.0  # % of float
        features['short_interest_change'] = 0.0  # Change vs prior month
        features['analyst_rating_avg'] = 0.0  # 1-5 scale
        features['analyst_rating_change'] = 0.0  # Change vs prior quarter
        features['earnings_surprise_last'] = 0.0  # % surprise
        
        return features
    
    def _cross_sectional_features(self, df: pd.DataFrame) -> dict:
        """
        Relative strength, sector momentum, market beta.
        """
        features = {}
        
        # These would be computed at the portfolio level
        # Placeholder for now
        features['sector_momentum'] = 0.0
        features['sector_relative_strength'] = 0.0
        features['market_beta_60d'] = 1.0
        features['market_beta_252d'] = 1.0
        features['correlation_to_spy_60d'] = 0.5
        features['correlation_to_sector_60d'] = 0.7
        features['idiosyncratic_vol'] = 0.0
        features['size_factor'] = 0.0  # Log market cap
        features['liquidity_factor'] = 0.0  # Dollar volume rank
        features['volatility_factor'] = 0.0  # Vol rank
        
        return features
```

#### 1.2 Feature Selection & Importance
```python
# technic_v4/engine/feature_selection.py

class FeatureSelector:
    """
    Automated feature selection using multiple methods.
    """
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "ensemble",
        top_n: int = 50,
    ) -> List[str]:
        """
        Select top N features using specified method.
        """
        if method == "ensemble":
            # Combine multiple methods
            scores = {}
            scores['mutual_info'] = self._mutual_info_selection(X, y)
            scores['random_forest'] = self._random_forest_importance(X, y)
            scores['lasso'] = self._lasso_selection(X, y)
            scores['correlation'] = self._correlation_selection(X, y)
            
            # Average ranks across methods
            avg_ranks = self._average_ranks(scores)
            return avg_ranks[:top_n]
        
        elif method == "mutual_info":
            return self._mutual_info_selection(X, y)[:top_n]
        
        elif method == "random_forest":
            return self._random_forest_importance(X, y)[:top_n]
        
        elif method == "lasso":
            return self._lasso_selection(X, y)[:top_n]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X, y, random_state=42)
        return X.columns[np.argsort(mi_scores)[::-1]].tolist()
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        return X.columns[np.argsort(rf.feature_importances_)[::-1]].tolist()
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        from sklearn.linear_model import LassoCV
        lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
        lasso.fit(X, y)
        coef_abs = np.abs(lasso.coef_)
        return X.columns[np.argsort(coef_abs)[::-1]].tolist()
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        correlations = X.corrwith(y).abs()
        return correlations.sort_values(ascending=False).index.tolist()
    
    def _average_ranks(self, scores: dict) -> List[str]:
        """Average ranks across multiple scoring methods."""
        ranks = {}
        for method, features in scores.items():
            for rank, feature in enumerate(features):
                if feature not in ranks:
                    ranks[feature] = []
                ranks[feature].append(rank)
        
        avg_ranks = {f: np.mean(r) for f, r in ranks.items()}
        return sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])
```

### **Pillar 2: Deep Learning Models**

#### 2.1 LSTM for Time Series
```python
# technic_v4/engine/alpha_models/lstm_alpha.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LSTMAlphaModel(nn.Module):
    """
    LSTM model for sequential alpha prediction.
    Captures temporal dependencies that tree models miss.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out.squeeze()

class TimeSeriesDataset(Dataset):
    """Dataset for LSTM training."""
    
    def __init__(self, features: pd.DataFrame, targets: pd.Series, sequence_length: int = 20):
        self.features = features.values
        self.targets = targets.values
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

def train_lstm_model(
    train_features: pd.DataFrame,
    train_targets: pd.Series,
    val_features: pd.DataFrame,
    val_targets: pd.Series,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> LSTMAlphaModel:
    """Train LSTM model with early stopping."""
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_features, train_targets)
    val_dataset = TimeSeriesDataset(val_features, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LSTMAlphaModel(input_size=train_features.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.squeeze())
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/alpha/lstm_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('models/alpha/lstm_best.pth'))
    return model
```

#### 2.2 Transformer for Multi-Horizon Prediction
```python
# technic_v4/engine/alpha_models/transformer_alpha.py

class TransformerAlphaModel(nn.Module):
    """
    Transformer model for multi-horizon alpha prediction.
    Better at capturing long-range dependencies than LSTM.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_horizons: int = 2,  # 5d and 10d
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_horizons)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        x = self.embedding(x)
        x = self.transformer(x)
        # Take last timestep
        x = x[:, -1, :]
        x = self.dropout(x)
        out = self.fc(x)
        return out  # Shape: (batch, num_horizons)
```

### **Pillar 3: Advanced Ensemble Methods**

#### 3.1 Stacking Ensemble
```python
# technic_v4/engine/alpha_models/stacking_ensemble.py

class StackingEnsemble:
    """
    Stacking ensemble combining multiple base models with a meta-learner.
    """
    
    def __init__(self):
        # Base models
        self.base_models = {
            'lgbm': LGBMAlphaModel(),
            'xgb': XGBAlphaModel(),
            'lstm': None,  # Will be initialized with proper input size
            'transformer': None,
        }
        
        # Meta-learner (learns to combine base model predictions)
        self.meta_model = Ridge(alpha=1.0)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Train base models and meta-learner.
        """
        # Train base models
        base_predictions_train = {}
        base_predictions_val = {}
        
        for name, model in self.base_models.items():
            if name in ['lgbm', 'xgb']:
                model.fit(X_train, y_train)
                base_predictions_train[name] = model.predict(X_train)
                base_predictions_val[name] = model.predict(X_val)
            elif name == 'lstm':
                # Train LSTM
                model = train_lstm_model(X_train, y_train, X_val, y_val)
                self.base_models[name] = model
                # Get predictions
                base_predictions_train[name] = self._lstm_predict(model, X_train)
                base_predictions_val[name] = self._lstm_predict(model, X_val)
            elif name == 'transformer':
                # Train Transformer
                model = train_transformer_model(X_train, y_train, X_val, y_val)
                self.base_models[name] = model
                # Get predictions
                base_predictions_train[name] = self._transformer_predict(model, X_train)
                base_predictions_val[name] = self._transformer_predict(model, X_val)
        
        # Create meta-features (base model predictions)
        meta_X_train = pd.DataFrame(base_predictions_train)
        meta_X_val = pd.DataFrame(base_predictions_val)
        
        # Train meta-learner on validation set (to avoid overfitting)
        self.meta_model.fit(meta_X_val, y_val)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        """
        base_predictions = {}
        for name, model in self.base_models.items():
            if name in ['lgbm', 'xgb']:
                base_predictions[name] = model.predict(X)
            elif name == '
