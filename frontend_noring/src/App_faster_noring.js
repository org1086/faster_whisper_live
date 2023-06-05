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
    const [confirmedText, setConfirmedText] = useState([]);
    const [validatingText, setValidatingText] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [selectedLanguage, setSelectedLanguage] = useState('en-US');

    const supportedLanguages = { 'en-US': 'English', 'de-DE': 'German', 'fr-FR': 'French', 'es-ES': 'Spanish' }

    function handleDataReceived(data, isMove2NextChunk, isPhraseComplete) {
        console.log('data: ', data)
        console.log('isMove2NextChunk: ', isMove2NextChunk)
        console.log('isPhraseComplete: ', isPhraseComplete)

        if (isPhraseComplete) {
            setConfirmedText(old => [...old, data + "\n"])
            // clear interim text
            setValidatingText('')
        } 
        else if (isMove2NextChunk){
            setConfirmedText(old => [...old, data])
            // clear interim text
            setValidatingText('')
        }
        else {
            setValidatingText(data)
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
        setConfirmedText([])
        setValidatingText([])
        setIsRecording(true)

        speechToTextUtils.initRecording(
            getTranscriptionConfig(),
            handleDataReceived,
            (error) => {
                console.error('Error when transcribing', error);
                setIsRecording(false)
                // No further action needed, as stream already closes itself on error
            });
    }

    function onStop() {
        setIsRecording(false);
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
            { (confirmedText.length > 0 || validatingText.length > 0)?
            <div style={{display: 'flex',  justifyContent:'center', //alignItems:'center', 
                         height: '80vh', padding: '0px 50px 50px 50px', overflowY: 'auto', whiteSpace: 'pre-wrap'}}>
                {confirmedText} {validatingText}
            </div> :
            <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '10vh'}}>
                Empty Transcribed Text.
            </div>
        }
        
        </div>
    );
}

export default AppNew;
