import './App.css';
import { io } from "socket.io-client";
import { useEffect, useState } from 'react';
import RecordRTC, { StereoAudioRecorder } from 'recordrtc';
import { saveAs } from 'file-saver';
import { AudioWave } from './audioWave';

const socket = io("localhost:5003/");

let record;

function App() {

    let count = 1;

    const send = () => {
        socket.emit("send_message", "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");
    }

    const [text, setText] = useState([]);
    const [name, setName] = useState('');
    const [paragraph, setParagraph] = useState([]);
    const [isRecording, setIsRecording] = useState(false);
    const [timeChunk, setTimeChunk] = useState(10);
    const [audio, setAudio] = useState(null);
    const [error, setError] = useState('');

    useEffect(() => {
        socket.on("receive_stream",
            data => {
                if (!data.paragraph)
                    setText(prev => [...prev, data.text]);
                else {
                    setText([]);
                    setParagraph(prev => [...prev, data.paragraph]);
                }
            }
            //data => console.log(data)
        );

        // socket.on("receive_message",
        //     data => setMsg(prev => {
        //         return [...prev, data]
        //     })
        //     //data => console.log(data)
        // );

        return () => {
            socket.off("receive_stream");
            socket.off("receive_message");
        }
    }, []);

    const convertBlobToBase64 = async (blob) => { // blob data
        return await blobToBase64(blob);
    }

    const blobToBase64 = blob => new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onload = () => resolve(reader.result.substr(reader.result.indexOf(',') + 1));
        reader.onerror = error => reject(error);
    });

    function _base64ToArrayBuffer(base64) {
        var binary_string = window.atob(base64);
        var len = binary_string.length;
        var bytes = new Uint8Array(len);
        for (var i = 0; i < len; i++) {
            bytes[i] = binary_string.charCodeAt(i);
        }
        return bytes.buffer;
    }

    const startRecord = () => {

        if (!name) {
            setError('Nhập tên người nói trước khi bắt đầu');
            return;
        }
        else {
            setError('');
        }
        setIsRecording(true);
        refreshState();

        navigator.mediaDevices.getUserMedia({
            audio: true
        }).then(async function (stream) {
            record = RecordRTC(stream, {
                type: 'audio',
                //mimeType: 'audio/wav',
                sampleRate: 44100,
                // used by StereoAudioRecorder
                // the range 22050 to 96000.
                // let us force 16khz recording:
                desiredSampRate: 16000,

                // MediaStreamRecorder, StereoAudioRecorder, WebAssemblyRecorder
                // CanvasRecorder, GifRecorder, WhammyRecorder
                recorderType: StereoAudioRecorder,
                // Dialogflow / STT requires mono audio
                numberOfAudioChannels: 1,

                timeSlice: timeChunk,

                ondataavailable: async function (blob) {
                    const newB = await new Blob([blob]).arrayBuffer();

                    //const s = await newB.arrayBuffer();
                    //const arrayBuffer = await new Response(blob).arrayBuffer();
                    let b64 = await convertBlobToBase64(blob);
                    let arrayBuffer = _base64ToArrayBuffer(b64)
                    console.log(`[${count}]: `, newB);
                    socket.emit("stream", { blob: newB, count });
                    count++;
                },
            });

            record.startRecording();

        });
    }

    const stopRecord = () => {
        record.stopRecording(blob => {
            setAudio(blob);
            socket.emit("stream", { blob: null, count: 0 });
        });
        setIsRecording(false);
    }

    const refreshState = () => {
        setAudio(null);
        setParagraph([]);
        setText([]);
        count = 1;
    }

    const handleDownloadFile = () => {
        const currentText = text.join(' ');
        const breakLine = [...paragraph].map(item => item + "\r \n");
        let blob = new Blob([...breakLine, currentText], { type: "text/plain;charset=utf-8" });
        saveAs(blob, `${name}.txt`);
    }

    return (
        <div className="App">
            <button onClick={send}>TEST</button>
            <h1>Hỗ trợ thư ký cuộc họp</h1>
            <div style={{
                display: 'flex',
                gap: '1rem',
                justifyContent: 'flex-start',
                margin: '5rem 10rem',
                height: '100%'
            }}>
                <div style={{
                    flex: 4,
                    textAlign: 'left'
                }}>
                    <div style={{
                        border: '1px solid #dddddd',
                        padding: '1rem',
                        display: 'flex',
                        justifyContent: 'space-between',
                        // alignItems: 'center'
                    }}>
                        <button className={`btn btn-start ${isRecording && 'btn-disable'}`} onClick={startRecord} disabled={isRecording}>{isRecording ? 'Đang ghi' : 'Bắt đầu'}</button>
                        {isRecording && <AudioWave />}
                        {error && <span className='error'>{error}</span>}
                        <button className={`btn btn-stop ${!isRecording && 'btn-disable'}`} onClick={stopRecord} disabled={!isRecording}>Dừng</button></div>
                    <h3>Nội dung:</h3>
                    <div style={{
                        border: '1px solid #dddddd',
                        padding: '0 1rem',
                        maxHeight: '550px',
                        overflowY: 'auto'
                    }}
                    >
                        {paragraph.map((item, index) => <p key={index} style={{ marginBottom: '1rem' }}>[{index}]: {item}</p>)}
                        <p>{text.join(' ')}</p>
                    </div>

                </div>
                <div style={{
                    flex: 3
                }}>
                    <div style={{
                        border: '1px solid #dddddd',
                        padding: '1rem',
                        display: 'flex',
                        justifyContent: 'flex-start',
                        alignItems: 'center',
                        gap: '0.5rem'
                    }}
                    >
                        {/* <input value={timeChunk} onChange={(e) => setTimeChunk(Number(e.target.value))} disabled/> */}
                        <span style={{ fontSize: '14px' }}>Tên người phát biểu: </span>
                        <input value={name} placeholder="Nhập tên người nói" onChange={(e) => setName(e.target.value)} />
                    </div>
                    <h3>Download file</h3>
                    <div>
                        <table>
                            <thead>
                                <tr>
                                    <th>Audio</th>
                                    <th>File txt</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>{audio && <audio src={audio} controls />}</td>
                                    <td>{audio && <span className='download-text' onClick={handleDownloadFile}>{name && `${name}.txt`}</span>}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
