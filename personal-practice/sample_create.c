#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){
    if(argc!=6){
        printf("Usage: %s <height1> <width1> <height2> <width2> <output_file_name> \n", argv[0]);
        return 1;
    }
    FILE * fd= fopen(argv[5], "w");
    if (fd == NULL) {
        printf("Error opening file %s\n", argv[5]);
        return 1;
    }
    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    fprintf(fd, "%d %d\n", height, width);
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            fprintf(fd, "%d ", i*width+j);
        }
        fprintf(fd, "\n");
    }
    int height2 = atoi(argv[3]);
    int width2 = atoi(argv[4]);
    fprintf(fd, "%d %d\n", height2, width2);
    for(int i=0; i<height2; i++){
        for(int j=0; j<width2; j++){
            fprintf(fd, "%d ", i*width2+j);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
    return 0;
}