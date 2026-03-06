import cv2
from insightface.model_zoo import get_model

model = get_model("models/w600k_r50.onnx")
model.prepare(ctx_id=-1)

img = cv2.imread("data/casia_webface_extracted/0000045/001.jpg")
img = cv2.resize(img, (112,112))

emb = model.get_feat(img)

print(emb.shape)