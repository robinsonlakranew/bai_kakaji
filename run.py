from bottlecap_inspection.inspect import inspect_cap

path = ''

image_bgr = load_image(path, resize_width=CFG['resize_width'])

result = inspect_cap(image_bgr, early_exit=True)

print(result["status"])
print(bin(result["plc_word"]))


