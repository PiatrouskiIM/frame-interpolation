# film-in-pytorch
Implementation of FILM: ... in pytorch. Training code is missin

`
ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4
`

Where the options are as follows:

out_w is the width of the output rectangle
out_h is the height of the output rectangle
x and y specify the top left corner of the output rectangle (coordinates start at (0,0) in the top left corner of the input)


You can refer to the input image size with in_w and in_h as shown in this first example. The output width and height can also be used with out_w and out_h.

https://video.stackexchange.com/questions/4563/how-can-i-crop-a-video-with-ffmpeg



    # # ffmpeg -i "${INPUT_VIDEO}" -vf "select='gte(scene,0)',metadata=print:file=scores.txt" -an -f null -
