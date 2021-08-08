import taichi as ti

ti.init()

## Load the image:
input_file_name = input('Enter the input image file name: ')
input_image = ti.imread(input_file_name)

## Process the image:
image = ti.field(ti.u8, input_image.shape)
image.from_numpy(input_image)


@ti.kernel
def process():
    for i, j, k in image:
        image[i, j, k] = 255 - image[i, j, k]  # revert color


process()

## Save the image:
output_image = image.to_numpy()

ti.imshow(output_image)

output_file_name = input('Enter the image file name to save: ')
ti.imwrite(output_image, output_file_name)
