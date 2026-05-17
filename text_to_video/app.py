import os
import uuid
import time
import re
import io
import requests
import warnings
import whisper
import numpy as np
from PIL import Image, ImageOps
from gtts import gTTS
from moviepy.config import change_settings
from moviepy.editor import (
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip
)
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.audio.fx import all as afx
from meta_ai_api import MetaAI
from keybert import KeyBERT

# NEW: local text-to-video imports
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import threading
import pathlib

change_settings({
    "IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"
})

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Base directory (folder where app.py lives)
BASE_DIR = pathlib.Path(__file__).resolve().parent

# Use absolute directories under the project folder so Flask and the worker agree
VIDEO_DIR = str(BASE_DIR / "videos")
IMAGE_DIR = str(BASE_DIR / "images")
AUDIO_DIR = str(BASE_DIR / "audio")
ANIMATION_TEMP_DIR = str(BASE_DIR / "generated_clips")

FPS = 24
TARGET_W, TARGET_H = 1280, 720
PEXELS_API_KEY = "8WyHmBd0Edmd5grI44S1xWcWJivZgx26cQtShqROLqa9yiuFhhUAcCwi"
MAX_PEXELS_RESULTS = 15
MATCH_TOKEN_THRESHOLD = 1

# NEW: speed factor for TTS playback (values >1.0 make the voice faster)
SPEED_FACTOR = 2.2

# ---------------------------
# LOCAL text-to-video configuration (second part ONLY)
# ---------------------------
LOCAL_T2V_MODEL = "damo-vilab/text-to-video-ms-1.7b"
HF_CLIP_SECONDS = 2

ANIMATION_JOBS = {}

# lazy-loaded pipeline (so app starts fast, model loads on first use)
_t2v_pipe = None


def get_t2v_pipe():
    """
    Lazily initialize and return the local text-to-video pipeline.
    Uses CUDA + fp16 for your RTX 4060 8GB.
    """
    global _t2v_pipe
    if _t2v_pipe is None:
        print("🧩 Loading local text-to-video model:", LOCAL_T2V_MODEL)
        _t2v_pipe = DiffusionPipeline.from_pretrained(
            LOCAL_T2V_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        _t2v_pipe.to("cuda")
        try:
            _t2v_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        print("✅ Local text-to-video model loaded.")
    return _t2v_pipe


os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(ANIMATION_TEMP_DIR, exist_ok=True)

# NLP helper to derive more descriptive image search phrases from narration sentences
kw_model = KeyBERT()

# ---------------------------
# FLASK BACKEND SETUP
# ---------------------------
app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


n = 1
while True:
    file_name = f"video-{n}.mp4"
    file_path = os.path.join(VIDEO_DIR, file_name)
    if not os.path.exists(file_path):
        VIDEO_OUTPUT = file_path
        break
    n += 1

# ---------------------------
# HELPERS
# ---------------------------
def normalize_tokens(s):
    return [t for t in re.split(r'\W+', s.lower()) if t]


def download_image_bytes(url, timeout=10):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return io.BytesIO(r.content)


def _keyphrase_from_sentence(sentence):
    """
    Use KeyBERT to derive a descriptive keyphrase for image search.
    Falls back to the first few words if extraction fails.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return ""

    try:
        keywords = kw_model.extract_keywords(
            sentence,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=1
        )
        if keywords:
            return keywords[0][0]
    except Exception as e:
        print(f"KeyBERT failed for sentence '{sentence}': {e}")

    return " ".join(sentence.split()[:4])


def pairs_from_script(script):
    """
    Convert a script/description into a list of (caption, keyphrase) pairs.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script) if s.strip()]
    pairs = []
    for s in sentences:
        keyphrase = _keyphrase_from_sentence(s)
        pairs.append((s, keyphrase))
    return pairs


# ---------------------------
# VIDEO GENERATION HELPERS (LOCAL text-to-video + slideshow fallback)
# ---------------------------
def sanitize_filename(text):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', (text or "animation"))
    safe = safe.strip("_") or "animation"
    return safe[:50]


def generate_animation_clip(prompt, index):
    """
    Generate a single short clip using a LOCAL text-to-video model
    (damo-vilab/text-to-video-ms-1.7b via diffusers).

    FIXED: use PIL RGB frames directly so colors are preserved and
    avoid ghost / washed-out grayscale artifacts.
    """
    pipe = get_t2v_pipe()

    num_frames = 16       # ~2 seconds at 8 fps
    fps = 8
    steps = 25

    print(f"🎞 Generating local T2V clip {index+1} for prompt: {prompt!r}")
    result = pipe(
        prompt,
        num_frames=num_frames,
        num_inference_steps=steps,
        output_type="pil"   # get proper PIL.Image frames
    )

    # result.frames can be [PIL,...] or [[PIL,...]] depending on pipeline
    frames = result.frames
    if len(frames) > 0 and isinstance(frames[0], list):
        frames = frames[0]

    pil_frames = []
    for i, frame in enumerate(frames):
        # Ensure PIL
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.array(frame))
        # Ensure RGB (no alpha, no grayscale)
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        pil_frames.append(frame)

    clip_filename = f"{sanitize_filename(prompt)}_clip{index + 1}.mp4"
    clip_path = os.path.join(ANIMATION_TEMP_DIR, clip_filename)

    # Directly export RGB frames to video – keeps full color
    export_to_video(pil_frames, clip_path, fps=fps)
    print(f"✅ Saved local T2V clip to {clip_path}")
    return clip_path


def generate_long_video(prompt, total_duration=10, output_path=None):
    """
    Stitch multiple locally-generated clips into a longer animation saved under VIDEO_DIR.
    """
    if not prompt:
        raise ValueError("Prompt is required for animation generation.")

    clip_count = max(1, (int(total_duration) + HF_CLIP_SECONDS - 1) // HF_CLIP_SECONDS)
    clip_paths = []

    for idx in range(clip_count):
        clip_path = generate_animation_clip(prompt, idx)
        clip_paths.append(clip_path)

    if not clip_paths:
        raise RuntimeError("No animation clips were generated.")

    video_clips = [VideoFileClip(path) for path in clip_paths]

    if output_path is None:
        output_path = os.path.join(VIDEO_DIR, f"animation_{uuid.uuid4().hex}.mp4")

    final_clip = concatenate_videoclips(video_clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=FPS)

    # Cleanup
    try:
        final_clip.close()
    except Exception:
        pass

    for vc in video_clips:
        try:
            vc.close()
        except Exception:
            pass

    for path in clip_paths:
        try:
            os.remove(path)
        except Exception:
            pass

    return output_path


def letterbox_image_from_bytes(img_bytes, out_path, target_w=TARGET_W, target_h=TARGET_H):
    with Image.open(img_bytes) as img:
        img = img.convert("RGB")
        img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        background = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        x = (target_w - img.width) // 2
        y = (target_h - img.height) // 2
        background.paste(img, (x, y))
        background.save(out_path, format="JPEG", quality=92)


def _score_photo_match(photo_meta, keyphrase_tokens):
    if not photo_meta or not keyphrase_tokens:
        return 0

    searchable = " ".join([
        (photo_meta.get("alt") or "").lower(),
        (photo_meta.get("url") or "").lower(),
        (photo_meta.get("photographer") or "").lower()
    ])

    score = 0
    for token in keyphrase_tokens:
        if token in searchable:
            score += 2
        elif token.rstrip("s") in searchable:
            score += 1
    return score


def is_photo_match(photo_meta, keyphrase):
    tokens = normalize_tokens(keyphrase)
    if not tokens:
        return False
    return _score_photo_match(photo_meta, tokens) >= MATCH_TOKEN_THRESHOLD


# ---------------------------
# CAPTIONS (MetaAI) - non-interactive
# ---------------------------
def generate_captions_from_topic(topic, min_sentences=10, max_sentences=12):
    if not topic or not topic.strip():
        raise ValueError("Topic is required")

    ai = MetaAI()
    prompt = (
        "Generate a nested list of captions for a topic. The output must follow this format:\n"
        '[["Topic", "<topic name>"], [[<sentence1>, <image key phrase1>], [<sentence2>, <image key phrase2>], ...]].\n'
        "For the given topic, produce between {min_s} and {max_s} descriptive sentences (each 4–12 words) and a concise keyword phrase for each that represents an image to search for. "
        "Keep everything relevant and do not include any extra explanation. Use simple sentences suitable for subtitles.\n"
    ).format(min_s=min_sentences, max_s=max_sentences)

    full_prompt = f"{prompt}Topic: {topic.strip()}"
    print("🎬 Generating captions from MetaAI for topic:", topic)
    response = ai.prompt(message=full_prompt)
    text = response.get("message", "") if isinstance(response, dict) else str(response)
    print(f"📝 MetaAI raw response:\n{text}\n")

    pairs = re.findall(r'\["([^"]+?)"\s*,\s*"([^"]+?)"\]', text)
    filtered = [(c.strip(), k.strip()) for c, k in pairs if c.strip().lower() != "topic"]

    if not filtered:
        lines = [l.strip() for l in re.split(r'[\r\n]+', text) if l.strip()]
        fallback = []
        for line in lines:
            parts = [p.strip() for p in re.split(r'\s*[,–-]\s*', line) if p.strip()]
            if len(parts) >= 2:
                fallback.append((parts[0], parts[1]))
        if fallback:
            filtered = fallback

    if not filtered:
        raise RuntimeError("Failed to parse caption pairs from MetaAI response.")

    if len(filtered) > max_sentences:
        filtered = filtered[:max_sentences]
    return filtered


# ---------------------------
# IMAGE SCRAPE + VERIFY + PROCESS
# ---------------------------
def scrape_and_verify_images(pairs):
    headers = {"Authorization": PEXELS_API_KEY}
    saved_paths = []

    for idx, (caption, keyphrase) in enumerate(pairs, start=1):
        query = keyphrase.strip()
        success = False
        chosen_photo = None
        tried = 0

        try:
            url = f"https://api.pexels.com/v1/search?query={requests.utils.requote_uri(query)}&per_page={MAX_PEXELS_RESULTS}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            photos = data.get("photos", [])
        except Exception:
            photos = []

        tokens = normalize_tokens(keyphrase)
        if photos and tokens:
            best = max(photos, key=lambda p: _score_photo_match(p, tokens))
            best_score = _score_photo_match(best, tokens)
            tried = len(photos)
            if best_score >= MATCH_TOKEN_THRESHOLD:
                chosen_photo = best
            elif photos:
                chosen_photo = best
        elif photos:
            chosen_photo = photos[0]

        if chosen_photo:
            try:
                img_url = (
                    chosen_photo.get("src", {}).get("large2x")
                    or chosen_photo.get("src", {}).get("large")
                    or chosen_photo.get("src", {}).get("medium")
                )
                if not img_url:
                    raise ValueError("No src URL in chosen photo.")
                img_bytes = download_image_bytes(img_url)
                out_path = os.path.join(IMAGE_DIR, f"image-{idx}.jpg")
                letterbox_image_from_bytes(img_bytes, out_path)
                saved_paths.append(out_path)
                print(f"✅ Saved image-{idx}.jpg (query='{query}', tried={tried})")
                success = True
            except Exception as e:
                print(f"⚠ Download/process failed for '{query}': {e}")

        if not success:
            try:
                fb_url = f"https://picsum.photos/{TARGET_W}/{TARGET_H}?random={idx}"
                img_bytes = download_image_bytes(fb_url)
                out_path = os.path.join(IMAGE_DIR, f"fallback-{idx}.jpg")
                letterbox_image_from_bytes(img_bytes, out_path)
                saved_paths.append(out_path)
                print(f"⚠ Used fallback image for '{query}'")
            except Exception as e:
                raise RuntimeError(f"Fatal: cannot obtain fallback image for index {idx}: {e}")

    return saved_paths


# ---------------------------
# AUDIO & SUBTITLES
# ---------------------------
def generate_audio(pairs, lang="en"):
    audio_paths = []
    for i, (caption, _) in enumerate(pairs):
        path = os.path.join(AUDIO_DIR, f"audio_{i}.mp3")
        gTTS(text=caption, lang=lang).save(path)
        audio_paths.append(path)
        print(f"🎵 Generated audio: {os.path.basename(path)}")
    return audio_paths


def generate_timed_subtitles(audio_files):
    model = whisper.load_model("tiny")
    subtitles = []
    t_offset = 0.0

    for i, audio in enumerate(audio_files):
        clip = AudioFileClip(audio)

        try:
            sped_audio = afx.speedx(clip, factor=SPEED_FACTOR)
        except Exception:
            sped_audio = clip

        temp_sped_path = os.path.join(AUDIO_DIR, f"temp_sped_{i}_{uuid.uuid4().hex}.mp3")
        sped_audio.write_audiofile(temp_sped_path, logger=None)

        result = model.transcribe(temp_sped_path)
        full_text = (result.get("text") or "").strip()

        sped_duration = sped_audio.duration
        subtitles.append(((t_offset, t_offset + sped_duration), full_text))
        t_offset += sped_duration

        clip.close()
        sped_audio.close()

        try:
            os.remove(temp_sped_path)
        except Exception as e:
            print(f"⚠ Could not remove temp file {temp_sped_path}: {e}")

    return subtitles


# ---------------------------
# VIDEO CREATION
# ---------------------------
def create_video(images, audio_files, subtitles, output_path=None):
    if output_path is None:
        output_path = globals().get("VIDEO_OUTPUT")
        if not output_path:
            output_path = os.path.join(VIDEO_DIR, f"video_{uuid.uuid4().hex}.mp4")

    print(f"[create_video] images={len(images)} audio_files={len(audio_files)} subtitles={len(subtitles)} output={output_path}")
    clips = []
    audio_clips = []
    image_clips = []

    def make_sub(txt):
        w = TARGET_W - 120
        return TextClip(txt,
                        font="Arial",
                        fontsize=36,
                        color="white",
                        stroke_color="black",
                        stroke_width=2,
                        method="caption",
                        size=(w, None),
                        align="center")

    MIN_SEC = 3.5
    IMAGES_PER_CAPTION = 2

    if not images:
        raise RuntimeError("No images available to create video.")
    if not audio_files:
        raise RuntimeError("No audio files available to create video.")

    required_images = max(len(audio_files), len(audio_files) * IMAGES_PER_CAPTION)
    if len(images) < required_images:
        multiplier = (required_images + len(images) - 1) // len(images)
        images = (images * multiplier)[:required_images]

    grouped_images = []
    img_idx = 0
    for _ in range(len(audio_files)):
        remaining = len(images) - img_idx
        take = IMAGES_PER_CAPTION
        if remaining < take:
            images.extend(images[:take - remaining])
        group = images[img_idx: img_idx + take]
        grouped_images.append(group)
        img_idx += take

    if img_idx < len(images):
        leftover = images[img_idx:]
        grouped_images[-1].extend(leftover)

    for imgs_for_audio, aud_path in zip(grouped_images, audio_files):
        aclip = AudioFileClip(aud_path)

        try:
            sped_audio = afx.speedx(aclip, factor=SPEED_FACTOR)
        except Exception:
            sped_audio = aclip

        audio_clips.append(sped_audio)

        if not imgs_for_audio:
            imgs_for_audio = [images[0]] if images else []

        k = len(imgs_for_audio) if imgs_for_audio else 1
        target_duration = getattr(sped_audio, "duration", 0) or (MIN_SEC * k)
        per_img_dur = MIN_SEC
        if target_duration and target_duration > 0 and k > 0:
            per_img_dur = target_duration / k
            if per_img_dur < MIN_SEC:
                per_img_dur = MIN_SEC

        durations = [per_img_dur for _ in range(k)]
        if target_duration and k > 0:
            total_visual = sum(durations)
            delta = target_duration - total_visual
            if abs(delta) > 1e-3:
                durations[-1] = max(0.1, durations[-1] + delta)

        sub_clips = []
        for img_path, dur in zip(imgs_for_audio, durations):
            ic = ImageClip(img_path).set_duration(dur).set_fps(FPS)
            sub_clips.append(ic)
            image_clips.append(ic)

        group_clip = concatenate_videoclips(sub_clips, method="compose")

        if target_duration:
            group_clip = group_clip.set_duration(target_duration)

        try:
            group_clip = group_clip.set_audio(sped_audio)
        except Exception:
            try:
                sub_clips[0] = sub_clips[0].set_audio(sped_audio)
            except Exception:
                pass

        clips.append(group_clip)

    if not clips:
        raise RuntimeError("No clips were created — check inputs.")

    final_clip = concatenate_videoclips(clips, method="compose")

    if subtitles:
        try:
            sub_clip = SubtitlesClip(subtitles, make_sub)
            positioned_sub = sub_clip.set_position(("center", TARGET_H - 120))
            out = CompositeVideoClip([final_clip, positioned_sub])
            use_subs = True
        except Exception as e:
            print(f"[create_video] Subtitles failed: {e}. Continuing without subtitles.")
            out = final_clip
            use_subs = False
    else:
        out = final_clip
        use_subs = False

    temp_audio = os.path.join(VIDEO_DIR, f"temp_audio_{uuid.uuid4().hex}.mp4")
    out.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        temp_audiofile=temp_audio,
        remove_temp=True
    )

    try:
        out.close()
    except Exception:
        pass
    try:
        final_clip.close()
    except Exception:
        pass
    if subtitles and use_subs:
        try:
            sub_clip.close()
        except Exception:
            pass

    for ac in audio_clips:
        try:
            ac.close()
        except Exception:
            pass

    for ic in image_clips:
        try:
            ic.close()
        except Exception:
            pass

    print(f"[create_video] finished. output={output_path}")


def exit():
    print(f"⚠ Deleting temporary files...")
    for folder in [IMAGE_DIR, AUDIO_DIR]:
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            try:
                os.remove(path)
            except PermissionError:
                print(f"⚠ Skipping {path} — it's still in use.")
            except Exception as e:
                print(f"⚠ Could not delete {path}: {e}")
    print("✅ Temporary files cleanup attempted.")


# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.route("/api/generate-animation", methods=["POST"])
def generate_animation_api():
    data = request.get_json(force=True)
    description = (data.get("description") or "").strip()
    duration = data.get("duration", 10)

    try:
        duration = max(5, int(duration))
    except (ValueError, TypeError):
        duration = 10

    if not description:
        return jsonify({"success": False, "message": "Description is required"}), 400

    job_id = uuid.uuid4().hex
    unique_name = f"animation_{job_id}.mp4"
    output_path = os.path.join(VIDEO_DIR, unique_name)
    video_url = f"/videos/{unique_name}"

    ANIMATION_JOBS[job_id] = {
        "status": "queued",
        "progress": 5,
        "message": "Queued for generation...",
        "video_path": output_path,
        "video_url": video_url
    }

    def worker():
        try:
            final_path = None
            success = False

            # Try 1: LOCAL text-to-video
            try:
                ANIMATION_JOBS[job_id]["status"] = "generating"
                ANIMATION_JOBS[job_id]["progress"] = 20
                ANIMATION_JOBS[job_id]["message"] = "Generating local AI animation..."
                print(f"[Animation {job_id}] Using local text-to-video...")
                final_path = generate_long_video(description, total_duration=duration, output_path=output_path)
                if final_path and os.path.exists(final_path):
                    success = True
                    print(f"[Animation {job_id}] ✅ Local text-to-video succeeded")
            except Exception as hf_err:
                print(f"[Animation {job_id}] ⚠ Local text-to-video failed: {hf_err}")
                final_path = None

            # Try 2: Fallback to slideshow
            if not success:
                try:
                    ANIMATION_JOBS[job_id]["status"] = "generating"
                    ANIMATION_JOBS[job_id]["progress"] = 40
                    ANIMATION_JOBS[job_id]["message"] = "Using slideshow fallback..."
                    pairs = pairs_from_script(description)
                    ANIMATION_JOBS[job_id]["progress"] = 60
                    ANIMATION_JOBS[job_id]["message"] = "Scraping images..."
                    images = scrape_and_verify_images(pairs)
                    if not images:
                        raise RuntimeError("No images available.")
                    ANIMATION_JOBS[job_id]["progress"] = 80
                    ANIMATION_JOBS[job_id]["message"] = "Creating slideshow video..."
                    create_slideshow_from_images(images, output_path, per_image_duration=3.0)
                    final_path = output_path
                    success = True
                    print(f"[Animation {job_id}] ✅ Slideshow fallback succeeded")
                except Exception as fallback_err:
                    raise RuntimeError(f"All animation methods failed. Last error: {fallback_err}")

            if not final_path or not os.path.exists(final_path):
                raise RuntimeError("Video file was not created successfully.")

            file_size = os.path.getsize(final_path)
            if file_size == 0:
                raise RuntimeError("Video file is empty.")

            ANIMATION_JOBS[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Animation ready!",
                "video_path": final_path
            })

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            ANIMATION_JOBS[job_id]["status"] = "error"
            ANIMATION_JOBS[job_id]["progress"] = 100
            ANIMATION_JOBS[job_id]["message"] = f"Failed: {str(e)[:100]}"

    threading.Thread(target=worker, daemon=True).start()

    return jsonify({
        "success": True,
        "message": "Animation generation started.",
        "video_url": video_url,
        "job_id": job_id
    }), 202


@app.route("/api/animation-status/<job_id>", methods=["GET"])
def animation_status(job_id):
    job = ANIMATION_JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "message": "Job not found"}), 404
    return jsonify({
        "success": True,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "video_url": job.get("video_url")
    }), 200


@app.route("/api/animation-download/<job_id>", methods=["GET"])
def animation_download(job_id):
    job = ANIMATION_JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        return jsonify({"success": False, "message": "Video not ready"}), 404
    filename = os.path.basename(job.get("video_path"))
    return send_from_directory(VIDEO_DIR, filename, as_attachment=True)


@app.route("/api/generate-narration", methods=["POST"])
def generate_narration_api():
    data = request.get_json(force=True)
    script = (data.get("script") or "").strip()
    if not script:
        return jsonify({"success": False, "message": "Script is required"}), 400

    try:
        is_short_topic = (len(script) <= 60) or (len(script.split()) <= 3)
        if is_short_topic:
            print("INFO: treating input as topic -> asking MetaAI for 10-12 captions")
            try:
                pairs = generate_captions_from_topic(script, min_sentences=10, max_sentences=12)
            except Exception as e:
                print("WARN: MetaAI captions failed, falling back to pairs_from_script():", e)
                pairs = pairs_from_script(script)
        else:
            pairs = pairs_from_script(script)

        if not pairs:
            return jsonify({"success": False, "message": "No caption pairs generated."}), 500

        images = scrape_and_verify_images(pairs)

        if len(images) < len(pairs):
            needed = len(pairs) - len(images)
            start_idx = len(images) + 1
            for i in range(needed):
                idx = start_idx + i
                fb_url = f"https://picsum.photos/{TARGET_W}/{TARGET_H}?random={idx}"
                try:
                    img_bytes = download_image_bytes(fb_url)
                    out_path = os.path.join(IMAGE_DIR, f"fallback-{idx}.jpg")
                    letterbox_image_from_bytes(img_bytes, out_path)
                    images.append(out_path)
                except Exception as e:
                    print(f"ERROR creating fallback image #{idx}: {e}")

        audio_files = generate_audio(pairs)
        if len(audio_files) < len(pairs):
            for i in range(len(audio_files), len(pairs)):
                path = os.path.join(AUDIO_DIR, f"audio_{i}.mp3")
                try:
                    gTTS(text=pairs[i][0], lang="en").save(path)
                    audio_files.append(path)
                except Exception as e:
                    print(f"ERROR regenerating audio {i}: {e}")

        subtitles = generate_timed_subtitles(audio_files)

        if not images or not audio_files or not subtitles:
            return jsonify({"success": False, "message": "Failed to prepare media for video."}), 500

        create_video(images, audio_files, subtitles)

        video_url = url_for('serve_video', filename=os.path.basename(VIDEO_OUTPUT), _external=False)
        return jsonify({
            "success": True,
            "message": "Video generation finished.",
            "video_url": video_url
        }), 200

    except Exception as e:
        print("❌ Error during generation:", e)
        return jsonify({"success": False, "message": f"Generation failed: {e}"}), 500


@app.route('/videos/<path:filename>', methods=['GET'])
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename, as_attachment=False)


def create_slideshow_from_images(images, output_path, per_image_duration=3.0):
    if not images:
        raise RuntimeError("No images provided for slideshow.")

    clips = []
    image_clips = []

    for img in images:
        ic = ImageClip(img).set_duration(per_image_duration).set_fps(FPS)
        clips.append(ic)
        image_clips.append(ic)

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, fps=FPS, codec="libx264", audio=False, threads=4)

    try:
        final.close()
    except Exception:
        pass
    for ic in image_clips:
        try:
            ic.close()
        except Exception:
            pass

    print(f"[create_slideshow_from_images] finished. output={output_path}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask backend on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
