using UnityEngine;
using UnityEngine.Windows.Speech;
using System.Diagnostics;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class StartListening : MonoBehaviour
{
    private DictationRecognizer dictationRecognizer;
    private string lastRecognizedText = "";
    void Start()
    {
        dictationRecognizer = new DictationRecognizer();

        dictationRecognizer.DictationResult += (text, confidence) =>
        {
            UnityEngine.Debug.LogFormat("Dictation result: {0}", text);
            lastRecognizedText = text;
            // Send this text to your Python script
            //ExecutePythonScript(text);
        };
    }

    void Update()
    {
        // If NumPad8 is pressed, start dictation
        if (Input.GetKeyDown(KeyCode.Keypad8))
        {
            if (dictationRecognizer.Status != SpeechSystemStatus.Running)
            {
                dictationRecognizer.Start();
                UnityEngine.Debug.Log("Started Dictation");
            }
        }

        // If NumPad2 is pressed, stop dictation and execute the script if there's a recognized text
        if (Input.GetKeyDown(KeyCode.Keypad2))
        {
            if (dictationRecognizer.Status == SpeechSystemStatus.Running)
            {
                dictationRecognizer.Stop();
                UnityEngine.Debug.Log("Stopped Dictation");


                if (!string.IsNullOrEmpty(lastRecognizedText))
                {
                    UnityEngine.Debug.Log("sentence recorded " + lastRecognizedText);
                    ExecutePythonScript(lastRecognizedText);  // Execute the script with the last recognized text
                    lastRecognizedText = "";  // Clear the recognized text
                }
            }
        }
    }

    void ExecutePythonScript(string text)
    {
        // Prepare to execute the Python script
        //string pythonPath = @"D:\miniconda3\envs\dreamgaussian\python.exe";  // Updated path
        string pythonPath = @"D:\miniconda3\Scripts\conda.exe";

        string scriptPath = @"C:\Users\eoint\3dGaussianWithVoice\dreamgaussian\main.py";
        string savePathText = text.Replace(" ", "_"); // replace spaces with underscores for save_path
        //string args = $"-c \"{scriptPath} --config configs/text.yaml prompt='a photo of a {text}' save_path={savePathText}\"";
        ///////string args = $"{scriptPath} --config dreamgaussian/configs/text.yaml prompt=\"a photo of a {text}\" save_path={savePathText}";
        string args = $"run -n dreamgaussian python {scriptPath} --config dreamgaussian/configs/text.yaml prompt=\"a photo of a {text}\" save_path={savePathText}";



        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        using (Process process = Process.Start(psi))
        {
            process.OutputDataReceived += (sender, e) =>
            {
                UnityEngine.Debug.Log("Output from Python: " + e.Data);
            };

            process.BeginOutputReadLine();

            process.ErrorDataReceived += (sender, e) =>
            {
                UnityEngine.Debug.LogError("Error from Python: " + e.Data);
            };

            process.BeginErrorReadLine();

            process.WaitForExit();
        }
        // Call this after the script finishes
        LoadGeneratedMesh(text);
    }



    void LoadGeneratedMesh(string userSpeech)
    {
#if UNITY_EDITOR
        UnityEditor.AssetDatabase.Refresh();
#endif

        string meshName = userSpeech[0].ToString().ToUpper() + userSpeech.Substring(1); // Capitalize the first letter
        string meshPath = "Assets/" + meshName + "_mesh.obj";
        GameObject meshPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(meshPath);

        if (meshPrefab != null)
        {
            GameObject groundObject = GameObject.Find("Ground"); // Assuming your ground object is named "Ground"
            Vector3 instantiatePosition = groundObject.transform.position + new Vector3(0, 3, 0); // 1 unit above the ground

            Instantiate(meshPrefab, instantiatePosition, Quaternion.identity);
        }
        else
        {
            UnityEngine.Debug.LogError("Failed to load the mesh: " + meshPath);
        }
    }

    void OnDestroy()
    {
        dictationRecognizer.DictationResult -= (text, confidence) =>
        {
            UnityEngine.Debug.LogFormat("Dictation result: {0}", text);
        };

        if (dictationRecognizer.Status == SpeechSystemStatus.Running)
        {
            dictationRecognizer.Stop();
        }
        dictationRecognizer.Dispose();
    }
}
