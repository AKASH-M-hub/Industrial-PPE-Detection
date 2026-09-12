import os
from PIL import Image, ImageDraw

def create_favicon():
    size = (64, 64)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Rounded blue background
    draw.rounded_rectangle([0, 0, 63, 63], radius=14, fill=(14, 165, 233, 255))
    
    # Chin strap
    draw.arc([22, 28, 42, 50], start=0, end=180, fill=(226, 232, 240, 255), width=3)
    draw.rounded_rectangle([29, 46, 35, 49], radius=1, fill=(51, 65, 85, 255))
    
    # Helmet crown
    draw.pieslice([18, 16, 46, 44], start=180, end=360, fill=(239, 68, 68, 255))
    draw.chord([26, 17, 38, 32], start=180, end=360, fill=(248, 113, 113, 255))
    
    # Helmet brim
    draw.ellipse([13, 31, 51, 37], fill=(220, 38, 38, 255))
    draw.ellipse([14, 30, 50, 35], fill=(239, 68, 68, 255))
    
    # White cross
    draw.rounded_rectangle([30, 21, 34, 29], radius=1, fill=(255, 255, 255, 255))
    draw.rounded_rectangle([28, 23, 36, 27], radius=1, fill=(255, 255, 255, 255))
    
    # Save as .ico and .png
    img.save('favicon.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    img.save('favicon.png', format='PNG')
    print('favicon.ico and favicon.png generated.')

if __name__ == '__main__':
    create_favicon()
