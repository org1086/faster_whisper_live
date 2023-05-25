import React, { useState } from "react";

import "./App.css";
import speechToTextUtils from "./utility_transcribe";
// import TranscribeOutput from "./TranscribeOutput";
// import SettingsSections from "./SettingsSection";

// const useStyles = () => ({
//     root: {
//         display: 'flex',
//         flex: '1',
//         margin: '100px 0px 100px 0px',
//         alignItems: 'center',
//         textAlign: 'center',
//         flexDirection: 'column',
//     },
//     title: {
//         marginBottom: '20px',
//     },
//     settingsSection: {
//         marginBottom: '20px',
//     },
//     buttonsSection: {
//         marginBottom: '40px',
//     },
// });

const AppNew = () => {
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

    function handleDataReceived(data, isMove2NextChunk, isPhraseComplete, isFinal) {
        console.log('data: ', data)
        console.log('isMove2NextChunk: ', isMove2NextChunk)
        console.log('isPhraseComplete: ', isPhraseComplete)
        console.log('isFinal: ', isFinal)
        if (isPhraseComplete) {
            setTranscribedData(old => [...old, data + "\n"])
            // clear interim text
            setInterimTranscribedData('')
        } 
        else if (isMove2NextChunk){
            setTranscribedData(old => [...old, data])
            // clear interim text
            setInterimTranscribedData('')
        }
        else {
            setInterimTranscribedData(data)
        }

        // if (isFinal) {
        //     setInterimTranscribedData('')
        //     setTranscribedData(oldData => [...oldData, data])
        // } else {
        //     setInterimTranscribedData(data)
        //     setTranscribedData(oldData => [...oldData, data])
        // }
    }

    function onNewJobStarted(started) {
        console.log('new job started: ', started)
        if (started) {            
            setTranscribedData([])
            setIsRecording(true)
        }
    }

    function getTranscriptionConfig() {
        return {
            audio: {
                encoding: 'LINEAR16',
                sampleRateHertz: 16000,
                languageCode: selectedLanguage,
            },
            interimResults: true
        }
    }

    function onStart() {

        speechToTextUtils.initRecording(
            getTranscriptionConfig(),
            handleDataReceived,
            onNewJobStarted,
            (error) => {
                console.error('Error when transcribing', error);
                setIsRecording(false)
                // No further action needed, as stream already closes itself on error
            });
    }

    function onStop() {
        setIsRecording(false)
        // flushInterimData() // A safety net if Google's Speech API doesn't work as expected, i.e. always sends the final result
        speechToTextUtils.stopRecording();
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
            <div style={{display: 'flex', height: '10vh', justifyContent:'center', alignItems:'center'
                        }}>
                {!isRecording && <button onClick={onStart} >Start transcribing</button>}
                {isRecording && <button onClick={onStop} >Stop</button>} 
            </div>
            {/* <div>
                <TranscribeOutput transcribedText={transcribedData} interimTranscribedText={interimTranscribedData} />
            </div> */}
            {/* {interimTranscribedData.length > 0 ?<div>Interim Transcribed Text: {interimTranscribedData}</div>:<div>Empty Interim Transcribed Text</div>}
            <br></br> */}
            { transcribedData.length > 0 ?
            <div style={{display: 'flex',  justifyContent:'center', //alignItems:'center', 
                         height: '80vh', padding: '0px 50px 50px 50px', overflowY: 'auto', whiteSpace: 'pre-wrap'}}>
                {transcribedData} {interimTranscribedData}
            </div> :
            <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '10vh'}}>
                Empty Transcribed Text.
            </div> 
        }
        
        </div>
    );
}

export default AppNew;
