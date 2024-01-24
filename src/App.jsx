import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";
import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

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

function App() {
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    (async function () {
      let pipelineInstance = await SummarizationPipelineFactory.getInstance(
        (data) => {}
      );
      setLoaded(true);
      console.log(pipelineInstance);
    })();
  }, []);
  return (
    <>
      <div className={`loading ${loaded && "hide"}`}>
        <span>Loading</span>
      </div>
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
