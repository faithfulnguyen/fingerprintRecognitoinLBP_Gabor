package fingerprint;
import java.awt.print.Printable;
import java.io.File;
import java.util.ArrayList;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_videostab.GaussianMotionFilter;

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
import static org.bytedeco.javacpp.opencv_core.print;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.CV_GRAY2RGB;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
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
    public static void main_version_1(String[] args) {
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
    
    public static void main(String[] args) {
        Mat img = imread ("src/fingerprint/1_1.tif",opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        resize(img,img,new Size(120,120));
        
        Mat normailizedImg = normalizeSubWindow(img);
        imwrite("img.jpg", normailizedImg);
        Fingerprint fp = new Fingerprint();
        Mat[] ori = fp.localSmooth(normailizedImg);
        Mat smooth = fp.smoothedOrientation(ori[0], ori[1]);
        fp.pointCare(smooth, img);
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
    
    // http://biomisa.org/wp-content/uploads/2013/07/fp_10.pdf
    
    public Mat[] sobelDerivatives(Mat image){
    	int scale = 1;
    	int delta = 0;
    	Mat grad_x = new Mat(), grad_y = new Mat();
    	//Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

    	Sobel( image, grad_x, 0, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    	//convertScaleAbs( grad_x, abs_grad_x );

    	Sobel( image, grad_y, 0, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    	//convertScaleAbs( grad_y, abs_grad_y );
    	return new Mat[]{grad_x, grad_y} ;
    }
    
    public double blockOrientation( Mat blck_sbl_x, Mat blck_sbl_y){
    	double v_x = 0;
    	double v_y = 0;
        int i = blck_sbl_x.rows()/2;
        int j = blck_sbl_x.cols()/2;
        int w = blck_sbl_x.rows();
    	UByteRawIndexer idx_x = blck_sbl_x.createIndexer();
    	UByteRawIndexer idx_y = blck_sbl_y.createIndexer();
    	for(int u = i - w/2; u < i + w/2; u++){
            for(int v = j - w/2; v < j + w/2; v++){
                v_x += 2 * idx_x.get(u, v) * idx_y.get(u, v);
                v_y += idx_x.get(u, v)*idx_x.get(u, v) - idx_y.get(u, v)*idx_y.get(u, v);
            }
    	}
    	double theta = 0.5 * (Math.PI +  Math.atan(v_y/v_x));
    	return theta;
    }
    
    public Mat localOrientation(Mat img, Mat blck_sbl_x, Mat blck_sbl_y){
    	int w = 10;
    	Mat oriImg = new Mat(img.rows()/ w, img.cols()/ w, opencv_core.CV_64F);
    	DoubleRawIndexer idx = oriImg.createIndexer();
    	int ori_i = 0;
    	int ori_j = 0;
    	for(int i = 0; i < img.rows(); i += w){
            for(int j = 0; j < img.cols(); j += w){
                Mat sbX = blck_sbl_x.apply(new Rect(j, i, w, w));
                Mat sbY = blck_sbl_y.apply(new Rect(j, i, w, w));
                double ori = blockOrientation(sbX, sbY);
                idx.put(ori_i,ori_j, ori);
                ori_j ++;
            }
            ori_j = 0;
            ori_i ++;
    	}
        //print(oriImg);
    	return oriImg;
    }
    
    public Mat[] localSmooth(Mat img){
    	Mat[] sb = sobelDerivatives(img);
    	Mat ori = localOrientation(img, sb[0], sb[1]);
    	Mat sinY = ori.clone();
    	Mat cosX = ori.clone();
    	DoubleRawIndexer idx_o = ori.createIndexer();
    	DoubleRawIndexer idx_x = cosX.createIndexer();
    	DoubleRawIndexer idx_y = sinY.createIndexer();
    	for(int i = 0; i < ori.rows(); i++){
            for(int j = 0; j <ori.cols(); j++){
                idx_x.put(i, j, 2 * Math.cos(idx_o.get(i, j)));
                idx_y.put(i, j, 2 * Math.sin(idx_o.get(i, j)));
            }
    	}
    	return new Mat[]{ cosX, sinY};
    }

    public Mat smoothedOrientation(Mat cosX, Mat sinY){
        opencv_imgproc.GaussianBlur(cosX, cosX, new Size(3,3), 0);
        opencv_imgproc.GaussianBlur(sinY, sinY, new Size(3,3), 0);
        Mat smooth = cosX.clone();
        DoubleRawIndexer idx_s = smooth.createIndexer();
        DoubleRawIndexer idx_x = cosX.createIndexer();
        DoubleRawIndexer idx_y = sinY.createIndexer();
        for(int i = 0; i < smooth.rows(); i++){
            for(int j = 0; j < smooth.cols(); j++){
                double theta = 0.5 * Math.atan(idx_y.get(i, j) / idx_x.get(i,j));
                idx_s.put(i, j, theta);
            }
        }
        
    	return smooth;
    }
    public void pointCare(Mat smooth, Mat img){
    	Mat siglar = smooth.clone();
    	DoubleRawIndexer index = siglar.createIndexer();
       
        Mat rgb = new Mat();
        opencv_imgproc.cvtColor(img, rgb, CV_GRAY2RGB );
        UByteRawIndexer idxI = rgb.createIndexer();
        opencv_core.MatExpr d = Mat.zeros(new Size(smooth.rows() + 2, smooth.cols() + 2) , CV_64F);
        Mat zero = d.asMat();
        Mat border = zero.apply(new Rect(1, 1, smooth.rows(), smooth.cols()));
        smooth.copyTo(border);
        int w = 3;
    	for(int r = 0; r < smooth.rows(); r ++){
            for(int c = 0; c < smooth.cols(); c ++){
                Mat tmp = zero.apply(new Rect(c, r, w, w));
                double beta = calcNeighbors(tmp);
                index.put(r, c, beta);
            }
    	}
        
        for(int i = 0; i < siglar.rows(); i++){
            for(int j = 0; j < siglar.cols(); j++){
                if(index.get(i, j) > 0.5 && index.get(i, j) <= 0.51){
                    //System.out.println(i*10 + " " + j*10); 
                    for(int r = 0; r < 5; r++){
                        idxI.put(i*10 + r, j*10 + r, 0, 150);
                    idxI.put(i*10 + r, j*10 + r, 1, 0);
                    idxI.put(i*10 + r, j*10 + r, 2, 0);
                    }
                    
                }
            }
        }
        imwrite("img.jpg", rgb);
        
       //print(siglar);
    }
    
    public double calcNeighbors(Mat tmp){
        int i = tmp.rows() / 2;
        int j = tmp.cols() / 2;
        DoubleRawIndexer idx = tmp.createIndexer();
        double beta = 0;
        double pc = Math.abs(Math.sin(idx.get(i + 1, j - 1)) - Math.sin(idx.get(i + 1, j))); // O2 - O1
        beta += checkConditional(pc);
        
        pc = Math.abs(Math.sin(idx.get(i, j - 1)) - Math.sin(idx.get(i + 1, j - 1))); // O3 - O2
        beta += checkConditional(pc);
        
        pc = Math.abs(Math.sin(idx.get(i - 1, j - 1)) - Math.sin(idx.get(i, j - 1))); //04- O3
        beta += checkConditional(pc);
        
        pc = Math.abs(Math.sin(idx.get(i - 1, j)) - Math.sin(idx.get(i - 1, j - 1))); // 05 - 04
        beta += checkConditional(pc);
        
        pc = Math.abs(Math.sin(idx.get(i - 1, j + 1)) - Math.sin(idx.get(i - 1, j))); // 06 - 05
        beta += checkConditional(pc);
        
        pc = Math.abs(Math.sin(idx.get(i, j + 1)) - Math.sin(idx.get(i - 1, j + 1))); // 07 - 06
        beta += checkConditional(pc);
        
        pc = Math.abs(Math.sin(idx.get(i + 1, j + 1)) - Math.sin(idx.get(i, j + 1))); // 08 - 07
        beta += checkConditional(pc);
       
        beta /= ( 2 * CV_PI);
        return beta;   
        
    }
    
    public double checkConditional(double pc){
        double beta = 0;
        if(pc <= (CV_PI / 2) && pc > -1.0 * CV_PI / 2)
            beta = pc;
        else if(pc <= (-1.0 * CV_PI / 2 ))
            beta = pc; 
        else beta = pc - CV_PI;
        return beta;
    }

    
}