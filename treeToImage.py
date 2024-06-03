# Modified from https://github.com/KolijnWolfaardt/GenProcTrees/blob/master/GenProcTrees/image_writer.py

"""
The MIT License (MIT)

Copyright (c) 2018 Kolijn Wolfaardt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from PIL import Image, ImageDraw

def treeToImage(tree, imageSize = (1024, 1024), base_x = 512, base_y = 512, thickness_scale = 4, draw_scale = 500):
  img = Image.new("RGBA", imageSize, (255, 255, 255, 0))
  draw = ImageDraw.Draw(img)

  for _, branch in tree.branches.items():

    thickness = int(math.log(branch.thickness))

    if thickness < 1:
      thickness = 1
    draw.line(
      (
        branch.start_pos[0] * draw_scale + base_x, base_y - branch.start_pos[2] * draw_scale,
        branch.end_pos[0] * draw_scale + base_x, base_y - branch.end_pos[2] * draw_scale
      ),
      width = thickness * thickness_scale, fill = (0, 0, 0, 255)
    )
  return img