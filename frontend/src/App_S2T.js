import React, { useState } from "react";

import "./App.css";
import AudioTextStreamer from "./utility_streaming";
// import TranscribeOutput from "./TranscribeOutput";
// import SettingsSections from "./SettingsSection";

const AppS2T = () => {
    const [transcribedData, setTranscribedData] = useState([]);
    const [interimTranscribedData, setInterimTranscribedData] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [selectedLanguage, setSelectedLanguage] = useState('en-US');

    const supportedLanguages = { 'en-US': 'English', 'de-DE': 'German', 'fr-FR': 'French', 'es-ES': 'Spanish' }
    
    function flushInterimData() {
        if (interimTranscribedData !== '') {
            setInterimTranscribedData('')
            setTranscribedData(oldData => [...oldData, interimTranscribedData])
        }
    }

    function handleDataReceived(data, isFinal) {
        if (isFinal) {
            setInterimTranscribedData('')
            setTranscribedData(oldData => [...oldData, data])
        } else {
            setInterimTranscribedData(data)
        }
    }

    function onStart() {
        setTranscribedData([])
        setIsRecording(true)

        AudioTextStreamer.startRecording(
            handleDataReceived,
            (error) => {
                console.error('Error when transcribing', error);
                setIsRecording(false)
                // No further action needed, as stream already closes itself on error
            });
    }

    function onStop() {
        setIsRecording(false)
        flushInterimData() // A safety net if Google's Speech API doesn't work as expected, i.e. always sends the final result
        AudioTextStreamer.stopRecording();
    }

    return (
        <div>
            {/* <div className={classes.title}>
                <Typography variant="h3">
                    Your Transcription App <span role="img" aria-label="microphone-emoji">🎤</span>
                </Typography>
            </div>
            <div className={classes.settingsSection}>
                <SettingsSections possibleLanguages={supportedLanguages} selectedLanguage={selectedLanguage}
                    onLanguageChanged={setSelectedLanguage} />
            </div> */}
            <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '100vh'}}>
                {!isRecording && <button onClick={onStart} >Start transcribing</button>}
                {isRecording && <button onClick={onStop} >Stop</button>}
            </div>
            {/* <div>
                <TranscribeOutput transcribedText={transcribedData} interimTranscribedText={interimTranscribedData} />
            </div> */}
        </div>
    );
}

export default AppS2T;
