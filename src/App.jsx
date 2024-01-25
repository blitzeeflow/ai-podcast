import {
  pipeline,
  env,
  Tensor,
  AutoTokenizer,
  SpeechT5ForTextToSpeech,
  SpeechT5HifiGan,
} from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.1";
import { useEffect, useRef, useState } from "react";
import { encodeWAV } from "./utils";
import "./App.css";
import {
  Button,
  Col,
  Container,
  Modal,
  ModalBody,
  ModalHeader,
  Row,
  Spinner,
} from "reactstrap";

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;
let pipelineInstance;
// Define model factories
// Ensures only one model is created of each type
class PipelineFactory {
  static task = null;
  static model = null;

  // NOTE: instance stores a promise that resolves to the pipeline
  static instance = null;

  constructor(tokenizer, model) {
    this.tokenizer = tokenizer;
    this.model = model;
  }

  /**
   * Get pipeline instance
   * @param {*} progressCallback
   * @returns {Promise}
   */
  static getInstance(progressCallback = null) {
    if (this.task === null || this.model === null) {
      throw Error("Must set task and model");
    }
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, {
        progress_callback: progressCallback,
      });
    }

    return this.instance;
  }
}

class SummarizationPipelineFactory extends PipelineFactory {
  static task = "summarization";
  static model = "Xenova/distilbart-cnn-6-6";
}

class TextGenerationPipelineFactory extends PipelineFactory {
  static task = "text-generation";
  static model = "Xenova/phi-1_5_dev";
}

// Use the Singleton pattern to enable lazy construction of the pipeline.
class TextToSpeechPipeline {
  static BASE_URL =
    "https://huggingface.co/datasets/Xenova/cmu-arctic-xvectors-extracted/resolve/main/";

  static model_id = "Xenova/speecht5_tts";
  static vocoder_id = "Xenova/speecht5_hifigan";

  static tokenizer_instance = null;
  static model_instance = null;
  static vocoder_instance = null;

  static async getInstance(progress_callback = null) {
    if (this.tokenizer_instance === null) {
      this.tokenizer = AutoTokenizer.from_pretrained(this.model_id, {
        progress_callback,
      });
    }

    if (this.model_instance === null) {
      this.model_instance = SpeechT5ForTextToSpeech.from_pretrained(
        this.model_id,
        {
          quantized: false,
          progress_callback,
        }
      );
    }

    if (this.vocoder_instance === null) {
      this.vocoder_instance = SpeechT5HifiGan.from_pretrained(this.vocoder_id, {
        quantized: false,
        progress_callback,
      });
    }

    return new Promise(async (resolve, reject) => {
      const result = await Promise.all([
        this.tokenizer,
        this.model_instance,
        this.vocoder_instance,
      ]);
      self.postMessage({
        status: "ready",
      });
      resolve(result);
    });
  }

  static async getSpeakerEmbeddings(speaker_id) {
    // e.g., `cmu_us_awb_arctic-wav-arctic_a0001`
    const speaker_embeddings_url = `${this.BASE_URL}${speaker_id}.bin`;
    const speaker_embeddings = new Tensor(
      "float32",
      new Float32Array(
        await (await fetch(speaker_embeddings_url)).arrayBuffer()
      ),
      [1, 512]
    );
    return speaker_embeddings;
  }
}

// Mapping of cached speaker embeddings
const speaker_embeddings_cache = new Map();

// Listen for messages from the main thread
const generateMp3 = async (text) => {
  // Load the pipeline
  const [tokenizer, model, vocoder] = await TextToSpeechPipeline.getInstance(
    (x) => {
      // We also add a progress callback so that we can track model loading.
    }
  );

  // Tokenize the input
  const { input_ids } = tokenizer(text);
  const speaker_id = "cmu_us_slt_arctic-wav-arctic_a0001";
  // Load the speaker embeddings
  let speaker_embeddings = speaker_embeddings_cache.get(speaker_id);
  if (speaker_embeddings === undefined) {
    speaker_embeddings = await TextToSpeechPipeline.getSpeakerEmbeddings(
      speaker_id
    );
    speaker_embeddings_cache.set(speaker_id, speaker_embeddings);
  }

  // Generate the waveform
  const { waveform } = await model.generate_speech(
    input_ids,
    speaker_embeddings,
    { vocoder }
  );

  // Encode the waveform as a WAV file
  const wav = encodeWAV(waveform.data);
  const blob = new Blob([wav], { type: "audio/wav" });
  var blobUrl = URL.createObjectURL(blob);
  const audio = document.querySelector("#audio-player");
  audio.src = blobUrl;
  console.log("done");
};

function App() {
  const [loaded, setLoaded] = useState(false);
  const [modelTitle, setModalTitle] = useState("Loading");
  const [summariseText, setSummariseText] = useState("");
  const textareaRef = useRef();
  useEffect(() => {
    if (pipelineInstance) return;
    console.log("here");
    (async function () {
      pipelineInstance = await SummarizationPipelineFactory.getInstance(
        (data) => {}
      );
      setLoaded(true);
    })();
  }, []);
  return (
    <>
      <Modal isOpen={!loaded}>
        <ModalHeader className="flex-center">
          <span>{modelTitle}</span>
        </ModalHeader>
        <ModalBody className="flex-center">
          <Spinner>Loading...</Spinner>
        </ModalBody>
      </Modal>
      <Container fluid>
        <Row>
          <Col>
            <textarea
              ref={textareaRef}
              placeholder="paste text here"
            ></textarea>
          </Col>
        </Row>
        <Row>
          <Col>
            <Button
              color="primary"
              onClick={async () => {
                if (textareaRef.current) {
                  // setLoaded(false);
                  setModalTitle("Generating");
                  const text = textareaRef.current.value;
                  const output = await pipelineInstance(text, {
                    max_new_tokens: 250,
                  });
                  const summaryText = output[0].summary_text;
                  setSummariseText(summaryText);
                  await generateMp3(summaryText);
                  // setLoaded(true);
                }
              }}
            >
              Generate
            </Button>
          </Col>
        </Row>
        <Row>
          <Col>
            <div>{summariseText}</div>
          </Col>
        </Row>
        <Row>
          <Col>
            <audio controls id="audio-player"></audio>
          </Col>
        </Row>
      </Container>
    </>
  );
}
// await pipeline(data.text, {
//   ...data.generation,
//   callback_function: function (beams) {
//       const decodedText = pipeline.tokenizer.decode(beams[0].output_token_ids, {
//           skip_special_tokens: true,
//       })

//       self.postMessage({
//           type: 'update',
//           target: data.elementIdToUpdate,
//           data: decodedText.trim()
//       });
//   }
// })
export default App;
