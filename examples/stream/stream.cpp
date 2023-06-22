// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//

#include "common.h"
#include "common-sdl.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <queue>

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t) {
    int64_t sec = t/100;
    int64_t msec = t - sec*100;
    int64_t min = sec/60;
    sec = sec - min*60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int) min, (int) sec, (int) msec);

    return std::string(buf);
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                 arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                 arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                 arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        //else if (arg == "-del" || arg == "--delay-ms")      { params.delay_ms      = std::stoi(argv[++i]); }
        //else if (arg == "-cth" || arg == "--conf-thold")    { params.confidence_threshold  = std::stoi(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"  || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"  || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"  || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n",        params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "\n");
}

std::string cleanTranscription(const std::string &transcription) {
    std::string cleanedText = "";
    bool ignore = false;
    for (char ch : transcription) {
        if (ch == '(' || ch == '[') {
            ignore = true;
        } else if (ch == ')' || ch == ']') {
            ignore = false;
        } else if (!ignore) {
            cleanedText += ch;
        }
    }
    return cleanedText;
}

std::string cleanExcessiveNewlines(const std::string& input) {
    // Remove excessive newlines
    std::regex newlinesRegex("\\\\n+");
    std::string cleaned = std::regex_replace(input, newlinesRegex, "");

    // Trim whitespaces from the start
    auto start = std::find_if(cleaned.begin(), cleaned.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    });

    // Trim whitespaces from the end
    auto end = std::find_if(cleaned.rbegin(), cleaned.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base();

    // Return the trimmed string
    return (end <= start ? std::string() : std::string(start, end));
}

void sendTranscriptionsOverHTTP(const std::queue<std::string>& transcriptionQueue) {
  try {
        // Create a copy of the transcriptionQueue
        std::queue<std::string> queueCopy = transcriptionQueue;

        // Concatenate transcriptions into a single string
        std::string concatenatedTranscriptions;
        while (!queueCopy.empty()) {
            concatenatedTranscriptions += queueCopy.front();
            queueCopy.pop();
        }

        // Clean excessive newlines and create a JSON object
        std::string prompt = cleanExcessiveNewlines(concatenatedTranscriptions);

        // Check if the prompt is not an empty string
        if (!prompt.empty()) {
            json jsonData;
            jsonData["prompt"] = "### Instruction: " + prompt + "\n### Response:";
            jsonData["threads"] = 1;
            jsonData["n_predict"] = 512;
            jsonData["batch_size"] = 1000;
            jsonData["temperature"] = 0.1;

            // Convert the JSON object to a
            // Convert JSON data to string
            std::string jsonString = jsonData.dump(4);

            // Set up the HTTP client
            httplib::Client client("127.0.0.1", 8080); // Adjust the host and port as needed
            // Set the connection timeout to 5 minutes (300 seconds)
            client.set_connection_timeout(300);
            // Optionally, set the read timeout to 5 minutes as well
            client.set_read_timeout(300);
            // Set the request path
            std::string path = "/completion"; // Adjust the path as needed

            // Set the request headers
            httplib::Headers headers;
            headers.emplace("Content-Type", "application/json");

            printf("%s\n", jsonString.c_str());
            // Send the HTTP POST request
            auto response = client.Post(path.c_str(), headers, jsonString.c_str(), "application/json");

            // Check the response status
            if (response && response->status == 200) {
                // Process the response body
                std::string responseString = response->body;

                // Parse the response JSON
                json responseJson = json::parse(responseString);

                std::string content = responseJson["content"];
                int tokens_predicted = responseJson["tokens_predicted"];

                // Handle the response data as needed
                printf("Received response: %s\nTokens predicted: %i", content.c_str(), tokens_predicted);

            } else {
                // Failed to receive a valid response
                if (response) {
                    std::cout << "Request failed. Status code: " << response->status << std::endl;
                } else {
                    std::cout << "Request failed. No response received." << std::endl;
                }
            }
        } else {
            printf("Prompt is empty, not sending.\n");
        }
    } catch (const std::exception &e) {
        // Catch and print exception
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    } catch (...) {
        // Catch any other exceptions not derived from std::exception
        std::cerr << "An unknown exception occurred" << std::endl;
    }
}


int main(int argc, char** argv) {

    std::queue<std::string> transcriptionQueue;
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    const float confidence_threshold = 1.0f;
    const int delay_ms = 250;

    params.keep_ms = std::min(params.keep_ms, params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_len = (1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps = !use_vad;
    params.no_context |= use_vad;
    params.max_tokens = 0;

    printf("use_Vad: %d\n", use_vad);

    // init audio

    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    // whisper init

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context* ctx = whisper_init_from_file(params.model.c_str());

    std::vector<float> pcmf32(n_samples_30s, 0.0f);
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
            __func__,
            n_samples_step,
            float(n_samples_step) / WHISPER_SAMPLE_RATE,
            float(n_samples_len) / WHISPER_SAMPLE_RATE,
            float(n_samples_keep) / WHISPER_SAMPLE_RATE,
            params.n_threads,
            params.language.c_str(),
            params.translate ? "translate" : "transcribe",
            params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        }
        else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    printf("[Start speaking..]\n");
    fflush(stdout);

    auto t_last = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_remaining;
    bool carry_over_prompt = false;
    bool delay_started = false;
    std::chrono::high_resolution_clock::time_point delay_start_time;
    bool lock_segment = false;

    // main audio loop
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // process new audio

        if (!use_vad) {
            while (true) {
                audio.get(params.step_ms, pcmf32_new);

                if ((int)pcmf32_new.size() > 2 * n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }

                if ((int)pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new * sizeof(float));

            pcmf32_old = pcmf32;
        }
        else {
            const auto t_now = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            }
            else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            t_last = t_now;
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress = false;
            wparams.print_special = params.print_special;
            wparams.print_realtime = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate = params.translate;
            wparams.single_segment = !use_vad;
            wparams.max_tokens = params.max_tokens;
            wparams.language = params.language.c_str();
            wparams.n_threads = params.n_threads;

            wparams.audio_ctx = params.audio_ctx;
            wparams.speed_up = params.speed_up;

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens = params.no_context ? 0 : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            // print result;
            {
                if (!use_vad) {
                    printf("\33[2K\r");

                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());

                    printf("\33[2K\r");
                }
                else {
                    const int64_t t1 = (t_last - t_start).count() / 1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size() * 1000.0 / WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int)t0, (int)t1);
                    printf("\n");
                }

                std::string transcription = ""; 
                const int n_segments = whisper_full_n_segments(ctx);

                for (int i = 0; i < n_segments; ++i) {

                    const char* text = whisper_full_get_segment_text(ctx, i);
                    const float confidence = 2.0f;//whisper_full_get_segment_confidence(ctx, i);

                    //printf("### confidence %f ", confidence);
                    if (!lock_segment && confidence < confidence_threshold) {
                        lock_segment = true;
                        continue;
                    }

                    if (lock_segment) {
                        // Skip the segment if it doesn't meet the confidence threshold
                        continue;
                    }

                    if (params.no_timestamps) {
                        transcription += cleanTranscription(std::string(text));
                        printf("%s", text);
                        fflush(stdout);
                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    }
                    else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        printf("[%s --> %s]  %s\n", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);
                        transcription += cleanTranscription(std::string(text)) + "\n"; 

                        if (params.fname_out.length() > 0) {
                            fout << "[" << to_timestamp(t0) << " --> " << to_timestamp(t1) << "]  " << text << std::endl;
                        }
                    }
                }

                // Check if transcription is a valid sentence before adding to the queue
                if (!transcription.empty() && transcription.length() >= 5 && !std::all_of(transcription.begin(), transcription.end(), isspace)) {
                    transcriptionQueue.push(transcription);
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                if (!delay_started) {
                    delay_start_time = std::chrono::high_resolution_clock::now();
                    delay_started = true;
                    carry_over_prompt = false;
                    lock_segment = false;
                }
                else {
                    const auto current_time = std::chrono::high_resolution_clock::now();
                    const auto delay_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - delay_start_time).count();
                    if (delay_duration >= delay_ms) {
                        delay_started = false;

                        // keep part of the audio for next iteration to try to mitigate word boundary issues
                        pcmf32_remaining = std::vector<float>(pcmf32.begin() + n_samples_keep, pcmf32.end());
                        pcmf32_old = std::move(pcmf32_remaining);

                        // Add tokens of the last full length segment as the prompt
                        if (!params.no_context || carry_over_prompt) {
                            prompt_tokens.clear();
                            carry_over_prompt = false;
                            const int n_segments = whisper_full_n_segments(ctx);
                            for (int i = 0; i < n_segments; ++i) {
                                const int token_count = whisper_full_n_tokens(ctx, i);
                                for (int j = 0; j < token_count; ++j) {
                                    prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                                }
                            }

                            if (n_segments > 0) {
                                carry_over_prompt = true;
                            }
                        }
                    }
                }
            }

            fflush(stdout);
            sendTranscriptionsOverHTTP(transcriptionQueue);
            // Clearing the queue after sending
            std::queue<std::string> emptyQueue;
            transcriptionQueue.swap(emptyQueue);

        }
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
