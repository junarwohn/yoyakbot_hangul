rm -r extract/*;
rm -r thumbs/*;
ffmpeg -i src.mp4 -vf fps=2 extract/img%04d.jpg;
