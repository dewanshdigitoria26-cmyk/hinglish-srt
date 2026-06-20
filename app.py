import gradio as gr
import subprocess, os, uuid, re
from faster_whisper import WhisperModel

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEVANAGARI → HINGLISH CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

CONSONANT_MAP = {
    'क': 'k',  'ख': 'kh', 'ग': 'g',  'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh','ज': 'j',  'झ': 'jh', 'ञ': 'n',
    'ट': 't',  'ठ': 'th', 'ड': 'd',  'ढ': 'dh', 'ण': 'n',
    'ड़': 'r', 'ढ़': 'rh',
    'त': 't',  'थ': 'th', 'द': 'd',  'ध': 'dh', 'न': 'n',
    'प': 'p',  'फ': 'f',  'ब': 'b',  'भ': 'bh', 'म': 'm',
    'य': 'y',  'र': 'r',  'ल': 'l',  'व': 'v',  'ळ': 'l',
    'श': 'sh', 'ष': 'sh', 'स': 's',  'ह': 'h',
    'क़': 'q',  'ख़': 'kh', 'ग़': 'g',  'ज़': 'z',
    'फ़': 'f',  'य़': 'y',
}

MATRA_MAP = {
    'ा': 'aa', 'ि': 'i',  'ी': 'i',  'ु': 'u',  'ू': 'oo',
    'े': 'e',  'ै': 'ai', 'ो': 'o',  'ौ': 'au',
    'ृ': 'ri', 'ं': 'n',  'ँ': 'n',  'ः': '',
}

VOWEL_MAP = {
    'अ': 'a',  'आ': 'aa', 'इ': 'i',  'ई': 'i',  'उ': 'u',  'ऊ': 'oo',
    'ए': 'e',  'ऐ': 'ai', 'ओ': 'o',  'औ': 'au', 'ऋ': 'ri',
    'ऑ': 'o',  'ऍ': 'e',
}

SPECIAL_CASES = {
    'क्षमा': 'kshama', 'क्ष': 'ksh',   'त्र': 'tr',
    'ज्ञ': 'gya',      'श्र': 'shr',   'द्ध': 'ddh',
    'द्व': 'dv',       'न्ह': 'nh',    'म्ह': 'mh',
    'ल्ह': 'lh',       'त्त': 'tt',    'क्क': 'kk',
    'च्च': 'chch',     'ज्ज': 'jj',    'ल्ल': 'll',
    'न्न': 'nn',       'म्म': 'mm',    'स्स': 'ss',
    'र्': 'r',
}

WORD_MAP = {
    'vah': 'woh',      'vo': 'woh',       'yah': 'yeh',
    'yaha': 'yahan',   'vaha': 'wahan',   'vahan': 'wahan',
    'apa': 'aap',      'aap': 'aap',      'ham': 'hum',
    'hama': 'hum',     'mujhe': 'mujhe',  'tumhe': 'tumhe',
    'unhe': 'unhe',    'inhe': 'inhe',
    'karana': 'karna', 'karna': 'karna',  'jana': 'jaana',
    'aana': 'aana',    'dena': 'dena',    'lena': 'lena',
    'bolna': 'bolna',  'hona': 'hona',    'rehna': 'rehna',
    'sunna': 'sunna',  'dekhna': 'dekhna','samajhna': 'samajhna',
    'batana': 'batana','milna': 'milna',  'padhna': 'padhna',
    'khelna': 'khelna','likhna': 'likhna',
    'kara': 'kar',     'karo': 'karo',    'kiya': 'kiya',
    'kii': 'ki',       'hain': 'hain',    'hai': 'hai',
    'tha': 'tha',      'thi': 'thi',      'the': 'the',
    'raha': 'raha',    'rahi': 'rahi',    'rahe': 'rahe',
    'chahie': 'chahiye','chahiye': 'chahiye',
    'milegi': 'milegi','milega': 'milega','milenge': 'milenge',
    'hoga': 'hoga',    'hogi': 'hogi',    'honge': 'honge',
    'men': 'mein',     'mem': 'mein',     'mein': 'mein',
    'se': 'se',        'ko': 'ko',        'ne': 'ne',
    'par': 'par',      'ka': 'ka',        'ki': 'ki',   'ke': 'ke',
    'usaki': 'uski',   'usaka': 'uska',   'unaki': 'unki',
    'unaka': 'unka',   'apaki': 'apki',   'apaka': 'apka',
    'tumhari': 'tumhari','tumhara': 'tumhara','tumhare': 'tumhare',
    'hamari': 'hamari','hamara': 'hamara','hamare': 'hamare',
    'meri': 'meri',    'mera': 'mera',    'mere': 'mere',
    'teri': 'teri',    'tera': 'tera',    'tere': 'tere',
    'nahin': 'nahi',   'naheen': 'nahi',  'nahi': 'nahi',
    'thika': 'theek',  'theek': 'theek',  'sahi': 'sahi',
    'accha': 'accha',  'acha': 'accha',   'pakka': 'pakka',
    'bahut': 'bahut',  'bahuta': 'bahut', 'bahot': 'bahut',
    'sirf': 'sirf',    'abhi': 'abhi',
    'pahle': 'pehle',  'pehle': 'pehle',  'pahale': 'pehle',
    'baad': 'baad',    'phir': 'phir',
    'lekin': 'lekin',  'isliye': 'isliye','kyunki': 'kyunki',
    'toh': 'toh',      'tah': 'toh',
    'kyom': 'kyun',    'kyun': 'kyun',
    'kya': 'kya',      'kahan': 'kahan',  'kaha': 'kahan',
    'kitna': 'kitna',  'kitni': 'kitni',  'kitne': 'kitne',
    'zyada': 'zyada',  'jyada': 'zyada',
    'thoda': 'thoda',  'thodi': 'thodi',
    'kam': 'kam',      'jaldi': 'jaldi',  'dhire': 'dheere',
    'zaruri': 'zaroori','zaroori': 'zaroori',
    'sabhi': 'sabhi',  'sab': 'sab',      'kuch': 'kuch',
    'liye': 'liye',    'yaar': 'yaar',    'bhai': 'bhai',
    'kal': 'kal',      'pata': 'pata',    'baat': 'baat',
    'bhi': 'bhi',      'hi': 'hi',        'mat': 'mat',
    'peyment': 'payment', 'akaunt': 'account',
    'bijaness': 'business', 'onalain': 'online',
    'teem': 'team',    'mobail': 'mobile',
}

HALANT       = '्'
ANUSVARA     = 'ं'
CHANDRABINDU = 'ँ'
NUKTA        = '़'
LONG_A_MATRA = 'ा'


def parse_devanagari_word(word):
    if word in SPECIAL_CASES:
        return SPECIAL_CASES[word]

    for deva, rom in SPECIAL_CASES.items():
        if len(deva) > 1:
            word = word.replace(deva, '\x00' + rom + '\x00')

    chars = list(word)
    n     = len(chars)
    syls  = []
    i     = 0

    while i < n:
        ch = chars[i]

        if ch == '\x00':
            j = i + 1; buf = ''
            while j < n and chars[j] != '\x00':
                buf += chars[j]; j += 1
            syls.append(('C_pure', buf))
            i = j + 1; continue

        if ch in VOWEL_MAP:
            v = VOWEL_MAP[ch]
            if i + 1 < n and chars[i + 1] in (ANUSVARA, CHANDRABINDU):
                v += 'n'; i += 1
            syls.append(('V', v)); i += 1; continue

        if ch in CONSONANT_MAP or (
            i + 1 < n and chars[i + 1] == NUKTA and ch + NUKTA in CONSONANT_MAP
        ):
            if i + 1 < n and chars[i + 1] == NUKTA and ch + NUKTA in CONSONANT_MAP:
                rc = CONSONANT_MAP[ch + NUKTA]; i += 2
            else:
                rc = CONSONANT_MAP.get(ch, ch); i += 1

            if i < n and chars[i] == HALANT:
                syls.append(('C_pure', rc)); i += 1; continue

            if i < n and chars[i] in MATRA_MAP:
                mc = chars[i]; mv = MATRA_MAP[mc]; i += 1
                is_long_a = (mc == LONG_A_MATRA)
                if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
                    mv += 'n'; i += 1
                syls.append(('CV', rc, mv, is_long_a))
            else:
                if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
                    syls.append(('CV', rc, 'an', False)); i += 1
                else:
                    syls.append(('Ca', rc))
            continue

        if ch in (ANUSVARA, CHANDRABINDU):
            syls.append(('V', 'n')); i += 1; continue

        if ch == 'ः': i += 1; continue

        syls.append(('X', ch)); i += 1

    total = len(syls)
    out   = []

    for idx, syl in enumerate(syls):
        is_last = (idx == total - 1)
        stype   = syl[0]

        if stype == 'Ca':
            rc = syl[1]
            if is_last:
                out.append(rc)
            elif idx > 0 and idx + 1 < total and syls[idx + 1][0] in ('CV', 'C_pure'):
                out.append(rc)
            else:
                out.append(rc + 'a')

        elif stype == 'CV':
            rc, mv = syl[1], syl[2]
            is_long_a = syl[3] if len(syl) > 3 else False
            if is_last and is_long_a:
                mv = 'a'
            out.append(rc + mv)

        elif stype == 'C_pure':
            out.append(syl[1])

        else:
            out.append(syl[1])

    return ''.join(out)


def is_devanagari(text):
    return bool(re.search(r'[\u0900-\u097F]', text))


def devanagari_to_hinglish(text):
    tokens = re.split(r'([\u0900-\u097F]+)', text)
    out    = []

    for tok in tokens:
        if not tok:
            continue
        if not is_devanagari(tok):
            out.append(tok)
            continue

        deva_words  = tok.split()
        roman_words = []
        for dw in deva_words:
            roman = parse_devanagari_word(dw)
            rl    = roman.lower()
            if rl in WORD_MAP:
                roman = WORD_MAP[rl]
            roman_words.append(roman)

        out.append(' '.join(roman_words))

    return re.sub(r' {2,}', ' ', ''.join(out)).strip()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def fmt(s):
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02}:{m:02}:{sec:06.3f}".replace(".", ",")


_model      = None
_model_size = None


def get_model(size="medium"):
    global _model, _model_size
    if _model is None or _model_size != size:
        _model = WhisperModel(
            size,
            device="cpu",
            compute_type="int8",
            num_workers=2,
            cpu_threads=4,
        )
        _model_size = size
    return _model


# ─── Main Transcription Function ─────────────────────────────────────────────

def transcribe_audio(audio_path, words_per_line, model_size, progress=gr.Progress()):
    if audio_path is None:
        return None, "❌ Pehle audio/video file upload karo!"

    try:
        job_id         = str(uuid.uuid4())
        wav_path       = f"outputs/{job_id}.wav"
        out_path       = f"outputs/{job_id}.srt"
        words_per_line = int(words_per_line)

        progress(0.05, desc="Audio convert ho raha hai...")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ar", "16000", "-ac", "1",
             "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
             wav_path],
            capture_output=True
        )
        if r.returncode != 0:
            return None, f"❌ FFmpeg error: {r.stderr.decode()}"

        progress(0.15, desc=f"Whisper {model_size} model load ho raha hai...")
        model = get_model(model_size)

        progress(0.3, desc="Transcribe ho rahi hai... thoda time lagega")
        segments_gen, info = model.transcribe(
            wav_path,
            task="transcribe",
            language="hi",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
            word_timestamps=False,
            chunk_length=30,
            initial_prompt=(
                "यह एक हिंदी बातचीत है जिसमें कुछ अंग्रेज़ी शब्द जैसे payment, "
                "account, business, online, team भी आते हैं।"
            ),
        )

        progress(0.5, desc="Segments collect ho rahe hain...")
        raw_segments = []
        seg_id = 1
        for seg in segments_gen:
            raw = seg.text.strip()
            if not raw:
                continue
            raw_segments.append({
                "id":       seg_id,
                "start":    seg.start,
                "end":      seg.end,
                "raw_text": raw,
            })
            seg_id += 1

        if not raw_segments:
            return None, "❌ Koi speech detect nahi hui. Audio check karo ya volume badhao."

        progress(0.6, desc="Hinglish conversion ho rahi hai...")
        total = len(raw_segments)
        for i, seg in enumerate(raw_segments):
            if is_devanagari(seg["raw_text"]):
                seg["hinglish_text"] = devanagari_to_hinglish(seg["raw_text"])
            else:
                seg["hinglish_text"] = seg["raw_text"]

            if i % 10 == 0:
                frac = 0.6 + 0.3 * (i / max(total, 1))
                progress(frac, desc=f"Converting... ({i}/{total})")

        progress(0.92, desc="SRT ban rahi hai...")
        parts = []
        n     = 1
        for seg in raw_segments:
            t1, t2 = seg["start"], seg["end"]
            text   = seg["hinglish_text"].strip()
            if not text:
                continue

            words = text.split()
            if not words:
                continue

            groups = [
                words[i:i + words_per_line]
                for i in range(0, len(words), words_per_line)
            ]
            tpg = (t2 - t1) / max(len(groups), 1)

            for j, g in enumerate(groups):
                gs   = t1 + j * tpg
                ge   = gs + tpg
                line = ' '.join(g)
                parts.append(f"{n}\n{fmt(gs)} --> {fmt(ge)}\n{line}\n\n")
                n += 1

        if n == 1:
            return None, "❌ Conversion ke baad koi valid text nahi mila."

        srt = ''.join(parts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(srt)

        try:
            os.remove(wav_path)
        except Exception:
            pass

        progress(1.0, desc="Ho gaya!")
        return out_path, (
            f"✅ Ho gaya! {n-1} subtitle lines ready!\n"
            f"📊 {len(raw_segments)} segments processed\n"
            f"🎯 Model: Whisper {model_size} + Custom Hinglish Parser"
        )

    except Exception as e:
        import traceback
        return None, f"❌ Error:\n{str(e)}\n\n{traceback.format_exc()}"


# ─── Gradio UI ────────────────────────────────────────────────────────────────

css = """
footer { display: none !important; }
.gradio-container { max-width: 760px !important; margin: 0 auto !important; }
"""

with gr.Blocks(title="HinglishSRT", css=css) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:32px 0 20px">
      <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:16px">
        <div style="width:44px;height:44px;background:linear-gradient(135deg,#ff6b35,#ffb347);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:24px">🎙</div>
        <span style="font-size:20px;font-weight:900;color:#e8eaf0">Hinglish<span style="color:#ff6b35">SRT</span></span>
      </div>
      <div style="font-size:1.9rem;font-weight:900;background:linear-gradient(90deg,#ff6b35,#ffb347);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px">
        Hindi Audio → WhatsApp Hinglish Subtitles
      </div>
      <div style="color:#5a6280;font-size:.95rem">
        100% Free • No API Key • Local Processing
      </div>
    </div>
    """)

    gr.HTML("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px">
      <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;text-align:center">
        <div style="font-size:1.5rem;margin-bottom:6px">📁</div>
        <div style="font-weight:700;font-size:.82rem;color:#e8eaf0;margin-bottom:3px">Koi bhi File</div>
        <div style="color:#5a6280;font-size:.7rem">MP3 MP4 WAV OGG M4A WEBM FLAC MKV</div>
      </div>
      <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;text-align:center">
        <div style="font-size:1.5rem;margin-bottom:6px">🆓</div>
        <div style="font-weight:700;font-size:.82rem;color:#e8eaf0;margin-bottom:3px">100% Free</div>
        <div style="color:#5a6280;font-size:.7rem">No API key • No cost • Local only</div>
      </div>
      <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;text-align:center">
        <div style="font-size:1.5rem;margin-bottom:6px">💬</div>
        <div style="font-weight:700;font-size:.82rem;color:#e8eaf0;margin-bottom:3px">WhatsApp Style</div>
        <div style="color:#5a6280;font-size:.7rem">Natural Roman Hinglish output</div>
      </div>
    </div>
    """)

    file_input = gr.File(
        label="🎵 Audio / Video Upload Karo",
        file_types=[".mp3", ".mp4", ".wav", ".ogg", ".m4a", ".webm",
                    ".flac", ".mkv", ".aac", ".wma", ".mov", ".avi"],
        type="filepath"
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["small", "medium"],
            value="medium",
            label="🤖 Whisper Model (medium = best accuracy)",
        )
        words_slider = gr.Slider(
            minimum=1, maximum=12, value=6, step=1,
            label="📝 Words per subtitle line"
        )

    submit_btn = gr.Button("🎯 Hinglish Subtitles Banao", variant="primary", size="lg")

    status_box = gr.Textbox(
        label="Status",
        interactive=False,
        placeholder="File upload karo aur button dabao..."
    )

    output_file = gr.File(label="⬇️ SRT File Download Karo")

    submit_btn.click(
        fn=transcribe_audio,
        inputs=[file_input, words_slider, model_dropdown],
        outputs=[output_file, status_box],
    )

    gr.HTML("""
    <div style="background:#0d1117;border:1px solid rgba(255,107,53,0.2);border-radius:14px;padding:18px;margin-top:8px">
      <div style="font-weight:700;color:#ff6b35;margin-bottom:10px;font-size:.85rem">💡 Tips:</div>
      <div style="color:#7a849a;font-size:.75rem;line-height:2">
        ✅ <b style="color:#e8eaf0">Koi API key nahi chahiye</b> — bilkul free hai!<br>
        ✅ <b style="color:#e8eaf0">Clear audio</b> use karo — background noise se accuracy kam hoti hai<br>
        ✅ <b style="color:#e8eaf0">medium model</b> recommend hai — small se ~30% better accuracy milti hai<br>
        ⚠️ Medium model pehli baar ~1.4GB download karega — internet chahiye<br>
        ⚠️ Custom syllable parser hai — bahut accurate, lekin rare words mein thoda adjust kar lena
      </div>
    </div>
    """)

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px;color:#3a4260;font-size:.72rem">
        Powered by Whisper + Custom Hinglish Parser • 100% Free • No API Required
    </div>
    """)

demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("DM Sans"),
    ),
)
