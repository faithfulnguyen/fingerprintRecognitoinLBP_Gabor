/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fingerprint;
import java.io.File;
import java.util.ArrayList;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Rect;
import static org.bytedeco.javacpp.opencv_core.convertScaleAbs;
import org.bytedeco.javacpp.opencv_imgcodecs;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.Sobel;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.util.ArrayList;
import java.util.Arrays;
import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import static org.bytedeco.javacpp.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_64F;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.normalize;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.calcHist;
import static org.bytedeco.javacpp.opencv_imgproc.compareHist;
import static org.bytedeco.javacpp.opencv_imgproc.filter2D;
import static org.bytedeco.javacpp.opencv_imgproc.getGaborKernel;

/**
 *
 * @author nguyentrungtin
 */
public class Fingerprint {

    private ArrayList<ArrayList<opencv_core.Mat>> trainData;
    private ArrayList<ArrayList<opencv_core.Mat>> testData;
    public Fingerprint(){
        this.trainData= new ArrayList<ArrayList<opencv_core.Mat>>();
        this.testData= new ArrayList<ArrayList<opencv_core.Mat>>();
    }
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        File folder = new File("");
        Fingerprint fg = new Fingerprint();
        String fileName = folder.getAbsolutePath() + "/src/finger/";
        System.out.println("Starting segment fingerprint!");
        File[] listOfFiles = new File(fileName).listFiles();
        Arrays.sort(listOfFiles);
        for(int idx = 0; idx < listOfFiles.length/ 8 ; idx++){
            ArrayList<opencv_core.Mat> trt = new ArrayList<>();
            ArrayList<opencv_core.Mat> tst = new ArrayList<>();
            for (int i = 0; i < 8; i++){
                if (listOfFiles[i + idx * 8].getName().contains(".tif")){
                    String name =  listOfFiles[idx * 8 + i].getName();
                    opencv_core.Mat image = imread(fileName + "/" + name, opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                    resize(image, image, new opencv_core.Size(120, 120));
                    opencv_imgproc.equalizeHist(image, image);
                    Mat norlImg = normalizeSubWindow(image);
                    imwrite(name, norlImg);
                    Mat hist = fg.gaborSubWindow(norlImg);
                    if(i < 6){
                        trt.add(hist);
                    }else tst.add(hist);
                }
            }
            fg.testData.add(tst);
            fg.trainData.add(trt);
        }
        fg.matchFingerprint();
    }
    
    public static opencv_core.Mat normalizeSubWindow(opencv_core.Mat image){
        int ut = 139;
        int vt = 100;//1607;  
        double u = meanMatrix(image);
        double v = variance(image, u);
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols() ; j++){
                double beta = Math.sqrt((vt * 1.0 / v ) * (Math.pow(idx.get(i, j) - u, 2)));
                if(idx.get(i, j) > ut){
                    idx.put(i, j, 0, (int)ut + (int)beta);
                }
                else idx.put(i, j, 0, Math.abs((int)ut - (int)beta));      
            }
        }
        return image;
    }
   
    public static double variance(opencv_core.Mat image, double mean){
        double var = 0; 
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols(); j++){
                var += Math.pow((idx.get(i, j) - mean), 2);
            }
        }
        var /= (image.cols() * image.rows());
        return var;
    }
    
    public static double meanMatrix(opencv_core.Mat img){
        double sum = 0;
        UByteRawIndexer idx = img.createIndexer();
        for(int i = 0; i < img.rows(); i++){
            for(int j = 0; j < img.cols(); j++){
                sum += idx.get(i, j, 0);
            }
        }
        sum /= (img.cols() * img.rows());
        return sum;
        
    }

    public opencv_core.Mat gaborSubWindow(Mat image){
        //resize(image, image, new opencv_core.Size(60, 60));
        int index = 0;
        double lm = 1, gm = 0.1, ps = CV_PI/4;
        double theta = 0;
        double[] sig = {1, 3, 5, 7, 9};
        //double sig = 7;
        int block = 20;
        int siz = image.cols() / block;
        Mat hist = new Mat(1, 5 * siz * siz * 60, CV_64F);
        DoubleRawIndexer dstIdx = hist.createIndexer();
        for(int r = 0; r < image.rows(); r += block){            
            for(int c = 0; c < image.cols(); c += block){
                for(int g = 0; g < 5; g++){
                    opencv_core.Mat kernel = getGaborKernel(new opencv_core.Size(4, 4), sig[g], theta, lm, gm, ps, CV_32F);
                    opencv_core.Mat gabor = new opencv_core.Mat(image.rows(), image.cols(), image.type());
                    Mat tmp = image.apply(new Rect(c, r, block, block)); 
                    filter2D(tmp, gabor, image.type(), kernel);  
                    Mat lbp = calcLBP(gabor);
                    Mat hs = calHistogramImage(lbp);
                    FloatRawIndexer hdx = hs.createIndexer();
                    for(int h = 0; h < hs.cols(); h++){
                        dstIdx.put(0, h + index, hdx.get(0, h));
                    }
                    index += 60;
                    theta += 15;
                }
                theta = 0;     
            }
        }      
        return hist;
    }

    public opencv_core.Mat gaborFilter(opencv_core.Mat image){       
        int siz = 30;
        int index = 0;
        //double lm = 4, gm = 0.08, ps = 90;
        double lm = 0.5 + 50 / 100.0, ps = 90, gm = 1;
        double theta = 45 * (180 / CV_PI);

        double[] sig = {1, 3, 5, 7, 9};
        opencv_core.Mat hist = new opencv_core.Mat(1, 5 * 10 * 256, CV_64F);
        DoubleRawIndexer dstIdx = hist.createIndexer();
        for(int i = 0; i < 5; i++){
            for(int j = 0; j < 10; j++){
                opencv_core.Mat gabor = new opencv_core.Mat(image.rows(), image.cols(), image.type());
                opencv_core.Mat kernel = getGaborKernel(new opencv_core.Size(3, 3), sig[i], theta , lm, gm, ps, CV_32F);
                filter2D(image, gabor, gabor.type(), kernel);
                Mat lbp = calcLBP(gabor);
                Mat hs = calHistogramImage(lbp);
                FloatRawIndexer hdx = hs.createIndexer();
                for(int h = 0; h < hs.cols(); h++){
                    dstIdx.put(0, h + index, hdx.get(0, h));
                }
                index += 256;
            }
        }	
        return hist;
    }
    
    public Mat calHistogramImage(Mat image){
        float[] range = { 0, 256 };
        int[] chanel = { 0 };
        int[] sz = { 60 };
        Boolean uniform = true; 
        Boolean accumulate = false;
        Mat hist = new Mat(256, 1, opencv_core.CV_64F);
        calcHist(image, 1, chanel, new Mat(), hist, 1, sz, range, uniform, accumulate );
        hist = hist.reshape(1, 1);
        normalize(hist, hist);
        return hist;
    }

    public static double chiSquare(Mat img, Mat img1){
        float dis = 0;
        DoubleRawIndexer idx = img.createIndexer();
        DoubleRawIndexer idx1 = img1.createIndexer();
        for(int i = 0; i < img.cols(); i++){
            if((idx.get(0, i) + idx1.get(0, i)) == 0){
               dis += 0;
            }
            else{
               dis += Math.pow((idx.get(0, i) - idx1.get(0, i)), 2)/(idx.get(0, i) + idx1.get(0, i));
            }
        }
        return dis;
   }
    
    public static Mat calcLBP(Mat image){
        Mat lbp = new Mat(image.rows() - 2, image.cols() - 2, image.type());
        UByteRawIndexer dst1Idx = image.createIndexer();
        UByteRawIndexer dst = lbp.createIndexer();
        int rows = image.rows(), cols = image.cols();
        for( int r = 1; r < rows - 1; r++){
            for( int c = 1; c < cols - 1; c++){
                float center = dst1Idx.get(r, c, 0);
                int code = 0;
                if(dst1Idx.get(r - 1, c - 1, 0) >= center){
                    code += 128;
                }
                if(dst1Idx.get(r - 1, c, 0) >= center){
                    code += 64;
                }
                if(dst1Idx.get(r - 1, c + 1, 0) >= center){
                    code += 32;
                }
                if(dst1Idx.get(r, c + 1, 0) >= center){
                    code += 16;
                }
                if(dst1Idx.get(r + 1, c + 1, 0) >= center){
                    code += 8;
                }
                if(dst1Idx.get(r + 1, c, 0) >= center){
                    code += 4;
                }
                if(dst1Idx.get(r + 1, c - 1, 0) >= center){
                    code += 2;
                }
                if(dst1Idx.get(r, c - 1, 0) >= center){
                    code += 1;
                }
                dst.put(r - 1, c - 1, code);
            }
        }
        return lbp;	
    }

    public double[] findClass(Mat hist){
        double[] score = new double[this.trainData.size()];
        for(int i = 0; i < this.trainData.size(); i++){
            double tmp = 0;
            for(int j = 0; j < this.trainData.get(i).size(); j++){
                //double dis = euclideanDistance(hist, this.trainData.get(i).get(j));
                //double dis = compareHist( hist, this.trainData.get(i).get(j), 0);
                double dis = chiSquare(hist, this.trainData.get(i).get(j));
                //double dis = baseProcess.Process.mahattan(hist, this.trainData.get(i).get(j));
                tmp += dis;
                //System.out.println(dis);
            }
            tmp = tmp / (this.trainData.get(0).size() * 1.0);
            score[i] = tmp; 
        }
        double[] min = new double[2];
        min[0] = 10000000;
        for (int i = 0; i < score.length; i++){
            if(min[0] > score[i]){
                min[0] = score[i];
                min[1] = i;
            }
        }
        return min;
    }
    
    public void matchFingerprint(){
        int err = 0;
        int[] label = new int[this.testData.size() * this.testData.get(0).size()];
        for(int i = 0; i < this.testData.size(); i++){
            for(int j = 0; j < this.testData.get(0).size(); j++){
                label[i* this.testData.get(0).size() + j] = i;
            }
        }
        for(int i = 0; i < this.testData.size(); i++){
            double[] a;
            for(int ele = 0; ele < this.testData.get(0).size(); ele++){
                a = this.findClass(this.testData.get(i).get(ele));
                if(a[1] != label[i * this.testData.get(0).size() + ele ])
                    err++;
                System.out.println( ele + ": " + "Predict: " + a[1] + " : " +  label[i * this.testData.get(0).size() + ele ] + " Distance: " + a[0]);
            }
            System.out.println(".......................");
        }
        System.out.println("Error : " + err + " Total: " + this.testData.size() *  this.testData.get(0).size());
        System.out.println( "Accuracy rate: " + (1 - ( err * 1.0) / (this.testData.size() * this.testData.get(0).size())));
    }
    
    public static double euclideanDistance(Mat img, Mat img1){
        double score = 0;
        DoubleRawIndexer idx = img.createIndexer();
        DoubleRawIndexer idx1 = img1.createIndexer();
        for(int i = 0; i < img.cols(); i++){
            score += Math.pow((idx.get(0, i) - idx1.get(0, i)), 2);
        }
        score = Math.sqrt(score);
        return score;
    }
    
    public static opencv_core.Mat orienImage(Mat image){
        int block = 8;
        opencv_core.Mat orient = new opencv_core.Mat(image.rows() - block / 2, image.cols() - block / 2, image.type());
        opencv_core.Mat grad_x = new opencv_core.Mat(), grad_y = new opencv_core.Mat();
        opencv_core.Mat abs_grad_x = new opencv_core.Mat(), abs_grad_y = new opencv_core.Mat();
        Sobel(image, grad_x, image.type(), 1, 0, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        Sobel(image, grad_y, image.type(), 0, 1, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );
        UByteRawIndexer dst = orient.createIndexer();
        for(int i = 0; i < image.rows() / block; i += block){
            for(int j = 0; j < image.cols()/  block; j += block){
                Mat dirX = abs_grad_x.apply(new Rect(i, j, block, block));
                Mat dirY = abs_grad_y.apply(new Rect(i, j, block, block));
                UByteRawIndexer ix = dirX.createIndexer();
                UByteRawIndexer iy = dirY.createIndexer();
                int x = dirX.rows() / 2;
                int y = dirX.cols() / 2;
                double ox = 0;
                double oy = 0;
                for(int r =  x - block / 2; r < x + block / 2; r++){
                    for(int c = y - block / 2; c < y + block / 2; c++){
                        ox += (2 * (ix.get(r, c) - iy.get(r, c)));
                        double tmp = ix.get(r, c) * ix.get(r, c);
                        double tmp1 = iy.get(r, c) * iy.get(r, c);
                        oy += (tmp - tmp1);
                    }
                }
                int ori = (int) ((int)0.5 * Math.atan(oy/ox));
                dst.put( i , j, ori);
                
            }
        }
        return orient;
    } 
    
}
