import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.*;
import org.opencv.videoio.Videoio;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


/**
 * @author Lycosa
 *
 */
public class detectAndDisplay {

     

    /** Global variables */
     private static String face_cascade_name = "haarcascade_frontalface_alt.xml";
     private static String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
     private static CascadeClassifier face_cascade = new CascadeClassifier();
     private static CascadeClassifier eyes_cascade = new CascadeClassifier();
     private static String window_name = "Capture - Face detection";


     public static void detectAndDisplay(Mat frame)
     {
       System.loadLibrary(Core.NATIVE_LIBRARY_NAME);  
       Mat frame_gray = new Mat();
       MatOfRect faces = new MatOfRect();

       Rect[] facesArray = faces.toArray();

       Imgproc.cvtColor(frame, frame_gray, Imgproc.COLOR_BGRA2GRAY);
       Imgproc.equalizeHist(frame_gray, frame_gray);

       //-- Detect faces
       face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, new Size(30, 30), new Size() );

       for (int i = 0; i < facesArray.length; i++)
       {
         Point center = new Point(facesArray[i].x + facesArray[i].width * 0.5, facesArray[i].y + facesArray[i].height * 0.5);
         Imgproc.ellipse(frame, center, new Size(facesArray[i].width * 0.5, facesArray[i].height * 0.5), 0, 0, 360, new Scalar(255, 0, 255), 4, 8, 0);

         Mat faceROI = frame_gray.submat(facesArray[i]);
         MatOfRect eyes = new MatOfRect();

         Rect[] eyesArray = eyes.toArray();

         //-- In each face, detect eyes
         eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0,new Size(30, 30), new Size());

         for (int j = 0; j < eyesArray.length; j++)
         {
            Point center1 = new Point(facesArray[i].x + eyesArray[i].x + eyesArray[i].width * 0.5, facesArray[i].y + eyesArray[i].y + eyesArray[i].height * 0.5);
            int radius = (int) Math.round((eyesArray[i].width + eyesArray[i].height) * 0.25);
            Imgproc.circle(frame, center1, radius, new Scalar(255, 0, 0), 4, 8, 0);
         }
       }
       //-- Show what you got
       Imgcodecs.imwrite(window_name, frame);
     }

    /**
     * @param args
     */
     public static void Main(String args[])
     {
       VideoCapture capture;
       Mat frame = new Mat();

       //-- 1. Load the cascades
       if (!face_cascade.load(face_cascade_name))
       {
           System.out.print("--(!)Error loading\n");
           return;
       }
       if (!eyes_cascade.load(eyes_cascade_name))
       {
           System.out.print("--(!)Error loading\n");
           return;
       }

       //-- 2. Read the video stream
       capture = new VideoCapture(0);
       if(!capture.isOpened())
       {
           System.out.println("Did not connect to camera.");
       }
       else
       {
           capture.retrieve(frame);
           detectAndDisplay(frame);
           capture.release();
       }
       }
}
     