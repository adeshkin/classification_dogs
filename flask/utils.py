def prepare_img(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_transform()['val']
    if transform:
        image = transform(image=img)["image"]

    return image.unsqueeze(0)