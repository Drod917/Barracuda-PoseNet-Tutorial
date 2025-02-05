﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawSkeleton : MonoBehaviour
{
    [Tooltip("The list of keypoint GameObjects that make up the pose skeleton")]
    public GameObject[] keypoints;

    // The GameObjects that contain data for the lines between keypoints
    private GameObject[] lines;

    // The line renderers that draw the lines between keypoints
    private LineRenderer[] lineRenderers;

    // The pairs of keypoints that should be connected on a body
    private int[][] jointPairs;

    // The width for the skeleton lines
    private float lineWidth = 5.0f;

    // Start is called before the first frame update
    void Start()
    {
        // The number of joint pairs
        int numPairs = keypoints.Length + 1;
        // Initalize the lines array
        lines = new GameObject[numPairs];
        // Initialize the lineRenderers array
        lineRenderers = new LineRenderer[numPairs];
        // Initialize the jointPairs array
        jointPairs = new int[numPairs][];

        // Initialize the pose skeleton
        initializeSkeleton();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void LateUpdate()
    {
        RenderSkeleton();
    }

    private void InitializeLine(int pairIndex, int startIndex, int endIndex, float width, Color color)
    {
        // Create a new joint pair with the specified start and end point indices
        jointPairs[pairIndex] = new int[] { startIndex, endIndex };

        // Create a new line GameObject
        string name = $"{keypoints[startIndex].name}_to_{keypoints[endIndex].name}";
        lines[pairIndex] = new GameObject(name);

        // Add LineRenderer component
        lineRenderers[pairIndex] = lines[pairIndex].AddComponent<LineRenderer>();
        // Set the material color
        lineRenderers[pairIndex].material.color = color;
        
        // The line will consist of two points
        lineRenderers[pairIndex].positionCount = 2;

        // Set the width from the start point
        lineRenderers[pairIndex].startWidth = width;
        // Set the width from the end point
        lineRenderers[pairIndex].endWidth = width;
    }

    private void initializeSkeleton()
    {
        // Nose to left eye
        InitializeLine(0, 0, 1, lineWidth, Color.magenta);
        // Nose to right eye
        InitializeLine(1, 0, 2, lineWidth, Color.magenta);
        // Left eye to left ear
        InitializeLine(2, 1, 3, lineWidth, Color.magenta);
        // Right eye to right ear
        InitializeLine(3, 2, 4, lineWidth, Color.magenta);

        // Left shoulder to right shoulder
        InitializeLine(4, 5, 6, lineWidth, Color.red);
        // Left shoulder to left hip
        InitializeLine(5, 5, 11, lineWidth, Color.red);
        // Right shoulder to right hip
        InitializeLine(6, 6, 12, lineWidth, Color.red);
        // Left shoulder to right hip
        InitializeLine(7, 5, 12, lineWidth, Color.red);
        // Right shoulder to left hip
        InitializeLine(8, 6, 11, lineWidth, Color.red);
        // Left hip to right hip
        InitializeLine(9, 11, 12, lineWidth, Color.red);

        // Left arm
        InitializeLine(10, 5, 7, lineWidth, Color.green);
        InitializeLine(11, 7, 9, lineWidth, Color.green);

        // Right arm
        InitializeLine(12, 6, 8, lineWidth, Color.green);
        InitializeLine(13, 8, 10, lineWidth, Color.green);

        // Left leg
        InitializeLine(14, 11, 13, lineWidth, Color.blue);
        InitializeLine(15, 13, 15, lineWidth, Color.blue);

        // Right leg
        InitializeLine(16, 12, 14, lineWidth, Color.blue);
        InitializeLine(17, 14, 16, lineWidth, Color.blue);
    }

    private void RenderSkeleton()
    {
        // Iterate through the joint pairs
        for (int i = 0; i < jointPairs.Length; i++)
        {
            // Set the start point index
            int startpointIndex = jointPairs[i][0];
            // Set the end point index
            int endpointIndex = jointPairs[i][1];

            // Set the GameObject for the starting key point
            GameObject startingKeyPoint = keypoints[startpointIndex];
            // Set the GameObject for the ending key point
            GameObject endingKeyPoint = keypoints[endpointIndex];

            // Get the starting position for the line
            Vector3 startPos = new Vector3(startingKeyPoint.transform.position.x,
                                           startingKeyPoint.transform.position.y,
                                           startingKeyPoint.transform.position.z);

            // Get the ending position for the line
            Vector3 endPos = new Vector3(endingKeyPoint.transform.position.x,
                                           endingKeyPoint.transform.position.y,
                                           endingKeyPoint.transform.position.z);

            // Check if both the starting and ending keypoints are active
            if (startingKeyPoint.activeInHierarchy && endingKeyPoint.activeInHierarchy)
            {
                // Activate the line
                lineRenderers[i].gameObject.SetActive(true);
                // Update the starting position
                lineRenderers[i].SetPosition(0, startPos);
                // Update the ending position
                lineRenderers[i].SetPosition(1, endPos);
            }
            else
            {
                // Deactivate the line
                lineRenderers[i].gameObject.SetActive(false);
            }
        }
    }
}
