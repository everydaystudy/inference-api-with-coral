from fastapi import FastAPI
import tflite_runtime.interpreter as tflite
import platform
import time
from PIL import Image
from PIL import ImageDraw
import detect

app = FastAPI()

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


label_file = "models/coco_labels.txt"
model_file = "models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"

labels = load_labels(label_file) if label_file else {}
interpreter = make_interpreter(model_file)
interpreter.allocate_tensors()


@app.get("/{item_id}/")
def read_root(item_id: str):
    image = Image.open("images/%s" % item_id)
    scale = detect.set_input(interpreter, image.size,
                            lambda size: image.resize(size, Image.ANTIALIAS))

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
            'loading the model into Edge TPU memory.')
    for _ in range(0, 5):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_output(interpreter, 0.4, scale)
        print('%.2f ms' % (inference_time * 1000))

    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    if "result.jpg":
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image.save("result.jpg")
        image.show()

    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}