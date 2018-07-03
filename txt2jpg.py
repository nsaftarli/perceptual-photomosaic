from PIL import ImageDraw,  ImageFont,  Image

char_array = ['M',  'N',  'H',  'Q',  '$',  'O',  'C',  '?',
              '7',  '>',  '!',  ':',  '-',  ';',  '.',  ' ']

char_dir = './assets/char_set/'
char_array = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '_', '+',
              '=', '~', 'Q',  '\\', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
              '[', ']', '{', '}', '|', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K',
              'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '/', '?', '!',
              '@', '#', '$', '%', '^', '&', '*', '(', ')',  ' ']
char_dir = "./assets/char_set_coloured_16b/"
print(len(char_array))
print(char_dir)

# char_array = [' ']

# char_array = ['/', '|', '\\', '-', 'L', 'O', '#', 'V', '.', ' ', ':', '~', '^', '!', '+', '=']
# char_dir = "./assets/char_set_2/"
for i, char in enumerate(char_array):
    canvas = Image.new("RGB",  (16, 16),  (0, 0, 0))

    draw = ImageDraw.Draw(canvas)
    myfont = ImageFont.truetype("./assets/fonts/RobotoMono-Regular.ttf", 16)
    draw.text((4, -4), char,  font=myfont,  fill=(255, 255, 255))
    # if i % 5 == 0:
    #   canvas.show()
    img_title = char_dir + str(i + (0 * 62)) + ".png"

    canvas.save(img_title,  "PNG")
