if [ "$#" -lt 1 ]; then
        echo "$# is Illegal number of parameters."
            echo "Usage: $0 [url]"
                exit 1
fi

args=("$@")
##echo "1 th parameter = ${args[0]}";
rm -f ../result-*;
rm -f src.mp4;
youtube-dl -f 137 ${args[0]} -o src.mp4;
rm -f -r extract/*;
rm -f -r thumbs/*;
ffmpeg -i src.mp4 -vf fps=2 extract/img%04d.jpg;
