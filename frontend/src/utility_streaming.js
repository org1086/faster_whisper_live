import io from 'socket.io-client';

import mic from 'microphone-stream';

const socket = new io.connect("http://localhost:5001/");

class AudioRecorder {
  constructor(){
    this.buffer = [];
    this.micStream = null;
  }

  initBuffer() {
    console.log("resetting buffer");
    this.buffer = [];
  }

  add2Buffer(raw) {
    this.buffer = this.buffer.concat(...raw);
    return this.buffer;
  }

  getDataFromBuffer() {
    return this.buffer;
  }

  setMicStream(stream) {
    this.micStream = stream;
  }
}

let count = 1;
let audioRecorder = new AudioRecorder();

let AudioTextStreamer = {

  /**
   * @param {function} onTranscribedData Callback each time it's received data event from socket server
   * @param {function} onError Callback  each time it's received error event from socket server
   */
   startRecording: function(onTranscribedData, onError) {

    console.log('start recording');
    audioRecorder.initBuffer();

    window.navigator.mediaDevices.getUserMedia({ video: false, audio: true }).then((stream) => {
      const startMic = new mic();

      startMic.setStream(stream);
      startMic.on('data', (chunk) => {
        var raw = mic.toRaw(chunk);
        if (raw == null) {
          return;
        }
        // audioRecorder.add2Buffer(raw);

      });

      audioRecorder.setMicStream(startMic);
    });

    // -------------------------------------------------------
    // socket-related functions
    if (onTranscribedData) {
      socket.on('speechData', (response) => {
        onTranscribedData(response.data, response.isFinal);
      });
    }
  },

  stopRecording: function() {

    socket.emit('stop');
    // stop and clear mic stream
    audioRecorder.micStream.stop();
    audioRecorder.setMicStream(null);
    
    console.log('stop recording');
  }  
};

export default AudioTextStreamer;

// Helper functions
/**
 * Converts a buffer from float32 to int16. Necessary for streaming.
 * sampleRateHertz of 1600.
 *
 * @param {object} buffer Buffer being converted
 */
function convertFloat32ToInt16(buffer) {
  let l = buffer.length;
  let buf = new Int16Array(l / 3);

  while (l--) {
    if (l % 3 === 0) {
      buf[l / 3] = buffer[l] * 0xFFFF;
    }
  }
  return buf.buffer
};