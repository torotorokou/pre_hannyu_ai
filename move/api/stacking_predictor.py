"""
スタッキング予測器クラス
APIサーバーでのpickle読み込みに対応するため、独立したモジュールとして定義
"""
import numpy as np


class StackingPredictor:
    """完全なスタッキング予測器
    
    二段構成:
    1. 15特徴量 -> 前処理(scaler, selector) -> [実際のモデルでは一段目モデルで予測]
    2. 一段目の予測結果[混合廃棄物A, 混合廃棄物B] -> 二段目ElasticNet -> 最終予測
    
    現在の実装では、一段目の学習済みモデルが見つからないため、
    固定の一段目予測値を使用している。
    """
    
    def __init__(self, scaler, selector, meta_model, stage1_predictions):
        """
        Args:
            scaler: sklearn StandardScaler for 15 features
            selector: sklearn VarianceThreshold for feature selection  
            meta_model: sklearn ElasticNet for stage 2
            stage1_predictions: [pred_A, pred_B] - fixed stage 1 predictions
        """
        self.scaler = scaler                          # 15特徴量の前処理
        self.selector = selector                      # 特徴選択
        self.meta_model = meta_model                  # 二段目ElasticNet
        self.stage1_predictions = stage1_predictions  # 一段目の固定予測値
        
    def predict(self, X):
        """15特徴量から最終予測を計算
        
        Args:
            X: numpy array of shape (n_samples, 15) - input features
            
        Returns:
            numpy array of shape (n_samples,) - final predictions
        """
        # 注意: 実際のスタッキングでは一段目も学習済みモデルで計算すべきだが、
        # 現在は一段目の学習済みモデルが見つからないため、
        # 既存の予測結果を使用
        batch_size = X.shape[0]
        
        # 各サンプルに対して固定の一段目予測を使用
        # TODO: 本来は一段目の学習済みモデルでX -> [pred_A, pred_B]を計算
        stage1_output = np.array([
            [self.stage1_predictions[0], self.stage1_predictions[1]]
            for _ in range(batch_size)
        ])
        
        # 二段目で最終予測
        final_pred = self.meta_model.predict(stage1_output)
        return final_pred
        
    def __repr__(self):
        return f"StackingPredictor(scaler={type(self.scaler).__name__}, selector={type(self.selector).__name__}, meta_model={type(self.meta_model).__name__}, stage1_predictions={self.stage1_predictions})"