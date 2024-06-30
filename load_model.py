from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np

# MobileNetV2ベースのモデルを定義
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# カスタムヘッドを追加
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)  # 出力は葉の数を表す1つのニューロン

model = Model(inputs=base_model.input, outputs=predictions)

# ベースモデルの層を凍結
for layer in base_model.layers:
    layer.trainable = False

# モデルのコンパイル
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])

# ダミーデータの生成
X_train = np.random.rand(100, 128, 128, 3)  # 100枚のランダムな画像データ
y_train = np.random.randint(0, 10, 100)     # 0から9のランダムなラベル

# モデルのトレーニング
model.fit(X_train, y_train, epochs=10)

# トレーニング済みモデルの保存
model.save('model/mobilenetv2_model.h5')
