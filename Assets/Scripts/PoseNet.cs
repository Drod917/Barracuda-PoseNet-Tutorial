using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using Unity.Barracuda;

public class PoseNet : MonoBehaviour
{
    [Tooltip("The input image that will be fed to the model")]
    // public RenderTexture videoTexture;
    public RenderTexture videoTexture;

    [Tooltip("The ComputeShader that will perform the model-specific preprocessing.")]
    public ComputeShader posenetShader;

    [Tooltip("The requested webcam height")]
    public int webcamHeight = 720;

    [Tooltip("The requested webcam width")]
    public int webcamWidth = 1280;

    [Tooltip("The requested webcam frame rate")]
    public int webcamFPS = 60;

    [Tooltip("The height of the image being fed to the model")]
    public int imageHeight = 360;
    
    [Tooltip("The width of the image being fed to the model")]
    public int imageWidth = 360;

    [Tooltip("Turn the InputScreen on or off")]
    public bool displayInput = false;

    [Tooltip("Use webcam feed as input")]
    public bool useWebcam = false;

    [Tooltip("The screen for viewing preprocessed images")]
    public GameObject inputScreen;

    [Tooltip("Stores the preprocessed image")]
    public RenderTexture inputTexture;

    [Tooltip("The model asset file to use when performing inference")]
    public NNModel modelAsset;

    [Tooltip("The backend to use when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("The minimum confidence level required to display the keypoint")]
    [Range(0, 100)]
    public int minConfidence = 70;

    [Tooltip("The list of keypoint GameObjects that make up the pose skeleton")]
    public GameObject[] keypoints;

    // The compiled model used for performing inference
    private Model m_RunTimeModel;

    // The interface used to execute the neural network
    private IWorker engine;

    // The name for the heatmap layer in the model asset
    private string heatmapLayer = "float_heatmaps";

    // The name for the offsets layer in the model asset
    private string offsetsLayer = "float_short_offsets";

    // The name for the Sigmoid layer that returns the heatmap predictions
    private string predictionLayer = "heatmap_predictions";

    // The number of keypoints estimated by the model
    private const int numKeypoints = 17;

    // Stores the current estimated 2D keypoint locations in the videoTexture
    // and their associated confidence values
    float[][] keypointLocations = new float[numKeypoints][];

    // Live video input from a webcam
    private WebCamTexture webcamTexture;

    // The height of the current video source
    private int videoHeight;

    // The width of the current video source
    private int videoWidth;

    void Start()
    {
        StartCoroutine(StartCo());
    }

    // Start is called before the first frame update
    IEnumerator StartCo()
    {
        // Get a reference to the Video Player
        GameObject videoPlayer = GameObject.Find("Video Player");

        // Get the Transform of the Video Screen
        Transform videoScreen = GameObject.Find("VideoScreen").transform;

        if (useWebcam)
        {
            // Create a new webcamTexture
            webcamTexture = new WebCamTexture();

            // Get the Transform component for the videoScreen GameObject
            // Transform videoScreen = GameObject.Find("VideoScreen").transform;

            // Flip the videoscreen around the y-axis
            videoScreen.rotation = Quaternion.Euler(0, 180, 0);
            // Invert the scale value for the z-axis
            videoScreen.localScale = new Vector3(videoScreen.localScale.x, videoScreen.localScale.y, -1f);

            // Start the camera
            webcamTexture.Play();

            // Deactivate the Video Player
            GameObject.Find("Video Player").SetActive(false);

            // Mac Fix
            yield return new WaitUntil(() => webcamTexture.height > 100);

            // Update the videoHeight
            videoHeight = (int)webcamTexture.height;

            // Update the videoWidth
            videoWidth = (int)webcamTexture.width;
        }
        else
        {
            // Update the videoHeight
            videoHeight = (int)videoPlayer.GetComponent<VideoPlayer>().height;
            // Update the videoWidth
            videoWidth = (int)videoPlayer.GetComponent<VideoPlayer>().width;
        }

        // Release the current videoTexture
        videoTexture.Release();

        // Create a new videoTexture using the current video dimensions
        videoTexture = new RenderTexture(videoWidth, videoHeight, 24, RenderTextureFormat.ARGB32);

        // Use a new videoTexture for Video Player
        videoPlayer.GetComponent<VideoPlayer>().targetTexture = videoTexture;

        // Apply the new videoTexture to the VideoScreen GameObject
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", videoTexture);

        // Adjust the videoScreen dimensions for the new videoTexture
        videoScreen.localScale = new Vector3(videoWidth, videoHeight, videoScreen.localScale.z);
        
        // Adjust the videoScreen position for the new videoTexture
        videoScreen.position = new Vector3(videoWidth / 2, videoHeight / 2, 1);

        // Get a reference to the main camera gameObject
        GameObject mainCamera = GameObject.Find("Main Camera");
        // Adjust the camera position to account for updates to the videoScreen
        mainCamera.transform.position = new Vector3(videoWidth / 2, videoHeight / 2, -(videoWidth / 2));
        // Adjust the camera size to account for updates to the videoScreen
        mainCamera.GetComponent<Camera>().orthographicSize = videoHeight / 2;

        // Compile the model asset into an object oriented representation
        m_RunTimeModel = ModelLoader.Load(modelAsset);

        // Create a model builder to modify the m_RunTimeModel
        var modelBuilder = new ModelBuilder(m_RunTimeModel);
        
        // Add the new Sigmoid layer that takes the output of the heatmap layer
        modelBuilder.Sigmoid(predictionLayer, heatmapLayer);

        // Create a worker that will execute the model with the selected backend
        engine = WorkerFactory.CreateWorker(workerType, modelBuilder.model);
    }

    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        if (useWebcam)
        {
            // Copy webcamTexture to videoTexture
            Graphics.Blit(webcamTexture, videoTexture);
        }
        // Preprocess the image for the current frame
        Texture2D processedImage = PreprocessImage();

        if (displayInput)
        {
            // Activate the InputScreen object
            inputScreen.SetActive(true);
            // Create a temporary Texture2D to store the rescaled input image
            Texture2D scaledInputImage = ScaleInputImage(processedImage);
            // Copy the data from the Texture2D to the RenderTexture
            Graphics.Blit(scaledInputImage, inputTexture);
            // GameObject.Find("InputScreen").GetComponent<RawImage>().texture = scaledInputImage;
            // Destroy the temporary Texture2D
            Destroy(scaledInputImage);
        }
        else
        {
            // Deactivate the InputScreen GameObject
            inputScreen.SetActive(false);
        }

        // Create a Tensor of shape [1, processedImage.height, processedImage.width, 3]
        Tensor input = new Tensor(processedImage, channels: 3);

        // Execute neural network with the provided input
        engine.Execute(input);

        // Determine the keypoint locations
        ProcessOutput(engine.PeekOutput(predictionLayer), engine.PeekOutput(offsetsLayer));

        // Update the positions of the keypoint GameObjects
        UpdateKeypointPositions();

        // Release GPU resources allocated for the Tensor
        input.Dispose();

        // Remove the processedImage variable
        Destroy(processedImage);
    }

    // Prepare the image to be fed into the neural network
    // Returns: The processed image
    private Texture2D PreprocessImage()
    {
        // Create a new Texture2D with the same dimensions as videoTexture
        Texture2D imageTexture = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGBA32, false);

        // Copy the RenderTexture contents to the new Texture2D
        Graphics.CopyTexture(videoTexture, imageTexture);

        // Make a temporary Texture2D to store the resized image
        Texture2D tempTex = Resize(imageTexture, imageHeight, imageWidth);

        // Remove the original imageTexture
        Destroy(imageTexture);

        // Apply model-specific preprocessing
        imageTexture = PreprocessResNet(tempTex);
        // Remove the temporary Texture2D
        Destroy(tempTex);

        return imageTexture;
    }

    // Resize the provided Texture2D
    // Returns: The resized image
    private Texture2D Resize(Texture2D image, int newWidth, int newHeight)
    {
        // Create a temporary RenderTexture
        RenderTexture rTex = RenderTexture.GetTemporary(newWidth, newHeight, 24);
        // Make the temporary RenderTexture the active RenderTexture
        RenderTexture.active = rTex; 

        // Copy the Texture2D to the temporary RenderTexture
        Graphics.Blit(image, rTex);
        // Create a new Texture2D with the new dimensions
        Texture2D nTex = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);

        // Copy the temporary RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);

        // Make the temporary RenderTexture not the active RenderTexture
        RenderTexture.active = null;

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(rTex);

        return nTex;
    }

    private Texture2D PreprocessResNet(Texture2D inputImage)
    {
        // Specify the numebr of threads on the GPU
        int numthreads = 8;

        // Get the index for the PreprocessResNet function in ComputeShader
        int kernelHandle = posenetShader.FindKernel("PreprocessResNet");

        // Define an HDR RenderTexture
        RenderTexture rTex = new RenderTexture(inputImage.width, inputImage.height, 24,
                                               RenderTextureFormat.ARGBHalf);

        // Enable random write access
        rTex.enableRandomWrite = true;
        // Create the HDR RenderTexture
        rTex.Create();

        // Set the value for the Result variable in ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", rTex);
        // Set the value for the inputImage variable in ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", inputImage);

        // Execute the ComputeShader
        posenetShader.Dispatch(kernelHandle, inputImage.width / numthreads,
                               inputImage.height / numthreads, 1);

        // Make the HDR RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Create a new HDR Texture2D
        Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

        // Copy the RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);

        // Make the HDR RenderTexture non-active
        RenderTexture.active = null;
        // Remove the HDR RenderTexture
        Destroy(rTex);

        return nTex; 
    }

    private Texture2D ScaleInputImage(Texture2D inputImage)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;

        // Get the index for the ScaleInputImage function in the ComputeShader
        int kernelHandle = posenetShader.FindKernel("ScaleInputImage");

        // Define an HDR RenderTexture
        RenderTexture rTex = new RenderTexture(inputImage.width, inputImage.height, 24,
                                               RenderTextureFormat.ARGBHalf);

        // Enable random write access
        rTex.enableRandomWrite = true;
        // Create the HDR RenderTexture
        rTex.Create();

        // Set the value for the Result variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", rTex);
        // Set the value for the InputImage variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", inputImage);

        // Execute the computeShader
        posenetShader.Dispatch(kernelHandle, inputImage.height / numthreads,
                               inputImage.width / numthreads, 1);

        // Make the HDR RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Create a new HDR Texture2D
        Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

        // Copy the RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);

        // Make the HDR RenderTexture non-active
        RenderTexture.active = null;

        // Remove the HDR RenderTexture
        Destroy(rTex);

        return nTex;
    }

    // Determine the estimated key point locations using the heatmaps and offset tensors
    private void ProcessOutput(Tensor heatmaps, Tensor offsets)
    {
        // Calculate the stride used to scale down the inputImage
        float stride = (imageHeight - 1) / (heatmaps.shape[1] - 1);
        stride -= (stride % 8);

        // The smallest dimension of the videoTexture
        int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);
        // The largest dimension of the videoTexture
        int maxDimension = Mathf.Max(videoTexture.width, videoTexture.height);

        // The value used to scale the key point locations up to the source resolution
        // float scale = (float)videoTexture.height / (float)imageHeight;
        float scale = (float)minDimension / (float)Mathf.Min(imageWidth, imageHeight);
        
        // The value used to compensate for resizing the source image to a square aspect ratio
        // float unsqueezeScale = (float)videoTexture.width / (float)videoTexture.height;
        float unsqueezeScale = (float)maxDimension / (float)minDimension;

        // Iterate through heatmaps
        for (int k = 0; k < numKeypoints; k++)
        {
            // Get the location of the current keypoint and its associated confidence value
            var locationInfo = LocateKeyPointIndex(heatmaps, offsets, k);

            // The (x,y) coordinates containing the confidence value in the current heatmap
            var coords = locationInfo.Item1;
            // The accompanying offset vector for the current coordinates
            var offset_vector = locationInfo.Item2;
            // The associated confidence values
            var confidenceValue = locationInfo.Item3;
            
            // Calculate the X-axis position
            // Scale the X coordinate up to the inputImage resolution
            // Add the offset vector to refine the keypoint location
            // Scale the position up to the videoTexture resolution
            // Compensate for any change in aspect ratio
            // float xPos = (coords[0]*stride + offset_vector[0])*scale*unsqueezeScale;
            float xPos = (coords[0]*stride + offset_vector[0])*scale;

            // Calculate the Y-axis position
            // Scale the Y coordinate up to the inputImage resolution and subtract it from the imageHeight
            // Add the offset vector to refine the keypoint location
            // Scale the position up to the videoTexture resolution
            float yPos = (imageHeight - (coords[1]*stride + offset_vector[1]))*scale;

            if (videoTexture.width > videoTexture.height)
            {
                xPos *= unsqueezeScale;
            }
            else 
            {
                yPos *= unsqueezeScale;
            }

            if (useWebcam)
            {
                xPos = videoTexture.width - xPos;
            }

            // Update the estimated keypoint location in the source image
            keypointLocations[k] = new float[] { xPos, yPos, confidenceValue };

            // DEBUG 
            // Debug.Log((keypointLocations[k][0], keypointLocations[k][1]));
        }
    }

    // Find the heatmap index that contains the highest confidence value and the associated offset vector
    private (float[], float[], float) LocateKeyPointIndex(Tensor heatmaps, Tensor offsets, int keypointIndex)
    {
        // Stores the highest confidence value found in the current heatmap
        float maxConfidence = 0f;

        // The (x,y) coordinates containing the confidence value in the current heatmap
        float[] coords = new float[2];
        // The accompanying offset vector for the current coordinates
        float[] offset_vector = new float[2];

        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.shape[1]; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.shape[2]; x++)
            {
                if (heatmaps[0, y, x, keypointIndex] > maxConfidence)
                {
                    // Update the highest confidence for the current keypoint
                    maxConfidence = heatmaps[0, y, x, keypointIndex];

                    // Update the estimated keypoint coordinates
                    coords = new float[] { x,y };

                    // Update the offset vector for the current keypoint location
                    offset_vector = new float[]
                    {
                        // X-axis offset
                        offsets[0, y, x, keypointIndex + numKeypoints],
                        // Y-axis offset
                        offsets[0, y, x, keypointIndex]
                    };
                }
            }
        }

        return ( coords, offset_vector, maxConfidence );
    }


    // Update the positions for the keypoint GameObjects
    private void UpdateKeypointPositions()
    {
        // Iterate through the keypoints
        for (int k = 0; k < numKeypoints; k++)
        {
            // Check if the current confidence value meets the confidence threshold
            if (keypointLocations[k][2] >= minConfidence / 100f)
            {
                // Activate the current keypoint GameObject
                keypoints[k].SetActive(true);
            }
            else
            {
                // Deactivate the current keypoint GameObject
                keypoints[k].SetActive(false);
            }

            // Create a new position Vector3
            // Set the z value to -1f to place it in front of the video screen
            Vector3 newPos = new Vector3(keypointLocations[k][0], keypointLocations[k][1], -1f);

            // Update the current keypoint location
            keypoints[k].transform.position = newPos;
        }
    }

}
