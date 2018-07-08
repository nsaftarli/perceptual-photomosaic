from PIL import ImageDraw,  ImageFont,  Image

# Creates templates out of ASCII characters


def txt2jpg(chars, out_dir, size, background, text):
    for i, char in enumerate(chars):
        canvas = Image.new('RGB',  (size, size),  (background, background, background))

        draw = ImageDraw.Draw(canvas)
        myfont = ImageFont.truetype('./data/fonts/RobotoMono-Regular.ttf', size)
        draw.text((4, -4), char,  font=myfont,  fill=(text, text, text))
        img_title = out_dir + str(i) + '.png'

        canvas.save(img_title,  'PNG')


if __name__ == '__main__':
    char_array = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '_', '+',
                  '=', '~', 'Q',  '\\', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
                  '[', ']', '{', '}', '|', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K',
                  'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '/', '?', '!',
                  '@', '#', '$', '%', '^', '&', '*', '(', ')',  ' ']
    out_dir = './assets/black_ascii_8/'
    size = 8
    background = 255
    text = 0
    txt2jpg(char_array, out_dir, size, background, text)
