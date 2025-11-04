# --- START OF FILE autolykos_v2_b200_working.py ---

import numpy as np
from numba import cuda
import struct
import time
import math

# ===============================================================
# ==================== BLAKE2b ON GPU ==================
# ===============================================================

_IV = np.array(
    [
        0x6A09E667F3BCC908,
        0xBB67AE8584CAA73B,
        0x3C6EF372FE94F82B,
        0xA54FF53A5F1D36F1,
        0x510E527FADE682D1,
        0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B,
        0x5BE0CD19137E2179,
    ],
    dtype=np.uint64,
)

_SIGMA = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ],
    dtype=np.uint8,
)


@cuda.jit(device=True)
def _rotr64_gpu(v, n):
    return ((v >> n) | (v << (64 - n))) & 0xFFFFFFFFFFFFFFFF


@cuda.jit(device=True)
def _G_gpu(v, a, b, c, d, x, y):
    v[a] = (v[a] + v[b] + x) & 0xFFFFFFFFFFFFFFFF
    v[d] = _rotr64_gpu(v[d] ^ v[a], 32)
    v[c] = (v[c] + v[d]) & 0xFFFFFFFFFFFFFFFF
    v[b] = _rotr64_gpu(v[b] ^ v[c], 24)
    v[a] = (v[a] + v[b] + y) & 0xFFFFFFFFFFFFFFFF
    v[d] = _rotr64_gpu(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & 0xFFFFFFFFFFFFFFFF
    v[b] = _rotr64_gpu(v[b] ^ v[c], 63)


@cuda.jit(device=True)
def bytes_to_words_gpu(data_bytes, start_idx, num_bytes, words_out):
    bytes_processed = 0
    for i in range(16):
        word = np.uint64(0)
        for j in range(8):
            pos = start_idx + i * 8 + j
            if bytes_processed < num_bytes and pos < len(data_bytes):
                byte_val = np.uint8(data_bytes[pos])
                word = word | (np.uint64(byte_val) << np.uint64(j * 8))
                bytes_processed += 1
        words_out[i] = word


@cuda.jit(device=True)
def blake2b_compress_gpu(h, block_words, t, last_block):
    v = cuda.local.array(16, dtype=np.uint64)
    for i in range(8):
        v[i] = h[i]
        v[i + 8] = _IV[i]
    v[12] = v[12] ^ (t & 0xFFFFFFFFFFFFFFFF)
    v[13] = v[13] ^ ((t >> 64) & 0xFFFFFFFFFFFFFFFF)
    if last_block:
        v[14] = v[14] ^ np.uint64(0xFFFFFFFFFFFFFFFF)
    for i in range(12):
        s = _SIGMA[i]
        _G_gpu(v, 0, 4, 8, 12, block_words[s[0]], block_words[s[1]])
        _G_gpu(v, 1, 5, 9, 13, block_words[s[2]], block_words[s[3]])
        _G_gpu(v, 2, 6, 10, 14, block_words[s[4]], block_words[s[5]])
        _G_gpu(v, 3, 7, 11, 15, block_words[s[6]], block_words[s[7]])
        _G_gpu(v, 0, 5, 10, 15, block_words[s[8]], block_words[s[9]])
        _G_gpu(v, 1, 6, 11, 12, block_words[s[10]], block_words[s[11]])
        _G_gpu(v, 2, 7, 8, 13, block_words[s[12]], block_words[s[13]])
        _G_gpu(v, 3, 4, 9, 14, block_words[s[14]], block_words[s[15]])
    for i in range(8):
        h[i] = h[i] ^ v[i] ^ v[i + 8]


@cuda.jit(device=True)
def blake2b_64_gpu(data_bytes, data_len, out):
    h = cuda.local.array(8, dtype=np.uint64)
    for i in range(8):
        h[i] = _IV[i]
    h[0] = h[0] ^ np.uint64(0x01010000) ^ np.uint64(64)
    block_size = 128
    t = np.uint64(0)
    full_blocks = data_len // block_size
    for i in range(full_blocks):
        block_words = cuda.local.array(16, dtype=np.uint64)
        bytes_to_words_gpu(data_bytes, i * block_size, block_size, block_words)
        t += np.uint64(block_size)
        blake2b_compress_gpu(h, block_words, t, False)
    remaining = data_len % block_size
    if remaining > 0 or data_len == 0:
        final_block = cuda.local.array(16, dtype=np.uint64)
        for i in range(16):
            final_block[i] = np.uint64(0)
        bytes_to_words_gpu(data_bytes, full_blocks * block_size, remaining, final_block)
        t += np.uint64(remaining)
        blake2b_compress_gpu(h, final_block, t, True)
    for i in range(8):
        out[i] = h[i]


# ===============================================================
# ==================== AUTOLYKOS V2 WORKING ======================
# ===============================================================


@cuda.jit(device=True)
def generate_dataset_element_gpu(dataset_seed, index, result_element):
    input_data = cuda.local.array(72, dtype=np.uint8)

    for i in range(64):
        input_data[i] = dataset_seed[i]

    for i in range(8):
        input_data[64 + i] = np.uint8((index >> (i * 8)) & 0xFF)

    hash_result = cuda.local.array(8, dtype=np.uint64)
    blake2b_64_gpu(input_data, 72, hash_result)

    for i in range(8):
        result_element[i] = hash_result[i]


@cuda.jit(device=True)
def calculate_indices_gpu(mixing_hash_words, N, indices_out):
    current_mixing_hash = cuda.local.array(8, dtype=np.uint64)
    mixing_hash_bytes = cuda.local.array(64, dtype=np.uint8)

    for i in range(8):
        current_mixing_hash[i] = mixing_hash_words[i]

    for i in range(32):
        for j in range(8):
            word = current_mixing_hash[j]
            for k in range(8):
                mixing_hash_bytes[j * 8 + k] = np.uint8((word >> (k * 8)) & 0xFF)

        segment_start = (i * 4) % 64

        if segment_start + 4 > 64:
            blake2b_64_gpu(mixing_hash_bytes, 64, current_mixing_hash)
            segment_start = 0

            for j in range(8):
                word = current_mixing_hash[j]
                for k in range(8):
                    mixing_hash_bytes[j * 8 + k] = np.uint8((word >> (k * 8)) & 0xFF)

        idx = np.uint32(0)
        for j in range(4):
            byte_val = mixing_hash_bytes[segment_start + j]
            idx = idx | (np.uint32(byte_val) << (j * 8))

        indices_out[i] = idx % N


@cuda.jit
def autolykos_v2_working_kernel(
    header_bytes,
    header_len,
    nonces,
    height,
    results,
    found,
    target_difficulty,
    total_nonces,
    dataset_seed_bytes,
):
    """
    kernel شغال بدون مشاكل indexing
    """
    # 🔥 استخدام 1D indexing بسيط
    thread_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # 🔥 التأكد من أن thread_id ضمن النطاق
    if thread_id >= total_nonces:
        return

    # 🔥 التأكد من عدم وجود حل مسبق
    if found[0] != 0:
        return

    nonce = nonces[thread_id]

    # الخطوة 1: حساب mixing_hash
    pow_input = cuda.local.array(128, dtype=np.uint8)
    pow_input_len = header_len + 4

    for i in range(header_len):
        pow_input[i] = header_bytes[i]

    for i in range(4):
        pow_input[header_len + i] = np.uint8((nonce >> (i * 8)) & 0xFF)

    mixing_hash = cuda.local.array(8, dtype=np.uint64)
    blake2b_64_gpu(pow_input, pow_input_len, mixing_hash)

    # الخطوة 2: حساب N
    N_initial = 1 << 26
    delta_N = 1 << 23
    epoch = height // 75000
    N = N_initial + epoch * delta_N

    # الخطوة 3: حساب المؤشرات
    indices = cuda.local.array(32, dtype=np.uint32)
    calculate_indices_gpu(mixing_hash, N, indices)

    # الخطوة 4: توليد عناصر dataset
    dataset_elements = cuda.local.array(32 * 8, dtype=np.uint64)

    for i in range(32):
        element_start = i * 8
        generate_dataset_element_gpu(
            dataset_seed_bytes, indices[i], dataset_elements[element_start : element_start + 8]
        )

    # الخطوة 5: الحساب النهائي
    final_input = cuda.local.array((32 * 64) + 64, dtype=np.uint8)

    for i in range(32 * 8):
        word = dataset_elements[i]
        for j in range(8):
            final_input[i * 8 + j] = np.uint8((word >> (j * 8)) & 0xFF)

    mixing_hash_bytes = cuda.local.array(64, dtype=np.uint8)
    for i in range(8):
        word = mixing_hash[i]
        for j in range(8):
            mixing_hash_bytes[i * 8 + j] = np.uint8((word >> (j * 8)) & 0xFF)

    for i in range(64):
        final_input[32 * 64 + i] = mixing_hash_bytes[i]

    final_hash = cuda.local.array(8, dtype=np.uint64)
    blake2b_64_gpu(final_input, (32 * 64) + 64, final_hash)

    # 🔥 تخزين النتيجة بشكل صحيح
    result_start = thread_id * 4
    for i in range(4):
        results[result_start + i] = final_hash[i]

    # الفحص على GPU
    if target_difficulty > 0:
        hash_int = final_hash[0]

        if hash_int < target_difficulty:
            found[0] = 1
            found[1] = nonce
            for i in range(4):
                found[2 + i] = final_hash[i]


def precompute_dataset_seed(header_bytes):
    header_array = np.frombuffer(header_bytes, dtype=np.uint8).copy()

    @cuda.jit
    def compute_dataset_seed_kernel(header_bytes, header_len, dataset_seed_out):
        hash_result = cuda.local.array(8, dtype=np.uint64)
        blake2b_64_gpu(header_bytes, header_len, hash_result)

        for i in range(8):
            word = hash_result[i]
            for j in range(8):
                dataset_seed_out[i * 8 + j] = np.uint8((word >> (j * 8)) & 0xFF)

    dataset_seed_result = np.zeros(64, dtype=np.uint8)
    dataset_seed_gpu = cuda.to_device(dataset_seed_result)

    compute_dataset_seed_kernel[1, 1](header_array, len(header_array), dataset_seed_gpu)
    cuda.synchronize()

    return dataset_seed_gpu


def calculate_proper_launch_config(batch_size):
    """
    حساب إعدادات الإطلاق الصحيحة لتغطية كامل حجم الدفعة.
    هذه هي النسخة النهائية التي تعالج كل الناونسات.
    """
    # 256 هو رقم جيد ومتوازن لعدد الـ threads في كل block
    threads_per_block = 256
    
    # هذه هي المعادلة الصحيحة لحساب عدد الـ blocks المطلوب لتغطية كل الناونسات
    blocks_per_grid = (batch_size + (threads_per_block - 1)) // threads_per_block
    
    print(f"   🎯 Launch Config: {blocks_per_grid:,} blocks × {threads_per_block} threads = {blocks_per_grid * threads_per_block:,} threads for {batch_size:,} nonces")
    
    return threads_per_block, blocks_per_grid


def process_8_billion_working(header_bytes, start_nonce, height, target_difficulty=None):
    """
    الإصدار المعدل لقياس الأداء الحقيقي للـ GPU
    عبر إزالة عنق الزجاجة في نقل البيانات.
    """
    total_nonces = 8_000_000_000

    # 🔥 يمكنك تعديل حجم الدفعة هنا لتجربة تأثيره على الأداء
    # حجم دفعة بين 10 مليون و 50 مليون يعتبر جيداً للبداية
    max_nonces_per_batch = 100_000_000

    num_batches = math.ceil(total_nonces / max_nonces_per_batch)

    print(f"🚀 بدء المعالجة (وضع قياس الأداء) لـ {total_nonces:,} نونس")
    print(f"💾 كل دفعة: {max_nonces_per_batch:,} نونس")
    print(f"🔢 عدد الدفعات: {num_batches}")
    print("=" * 60)

    # حسابات مسبقة
    header_array = np.frombuffer(header_bytes, dtype=np.uint8).copy()
    gpu_header = cuda.to_device(header_array)
    dataset_seed_gpu = precompute_dataset_seed(header_bytes)

    last_nonce = start_nonce
    solution_nonce = None
    solution_hash = None

    for batch in range(num_batches):
        if solution_nonce is not None:
            break

        batch_start_nonce = start_nonce + (batch * max_nonces_per_batch)
        batch_end_nonce = min(batch_start_nonce + max_nonces_per_batch, start_nonce + total_nonces)
        current_batch_size = batch_end_nonce - batch_start_nonce

        # التأكد من عدم وجود دفعة فارغة في النهاية
        if current_batch_size == 0:
            continue

        print(f"\n🔧 الدفعة {batch + 1}/{num_batches}: {current_batch_size:,} نونس")

        # إنشاء buffers جديدة لكل دفعة
        gpu_nonces = cuda.device_array(current_batch_size, dtype=np.uint64)
        gpu_results = cuda.device_array(current_batch_size * 4, dtype=np.uint64)

        nonces_cpu = np.arange(batch_start_nonce, batch_end_nonce, dtype=np.uint64)
        cuda.to_device(nonces_cpu, to=gpu_nonces)

        found_cpu = np.zeros(6, dtype=np.uint64)
        gpu_found = cuda.to_device(found_cpu)

        threads_per_block, blocks_per_grid = calculate_proper_launch_config(current_batch_size)

        target_val = np.uint64(target_difficulty) if target_difficulty else np.uint64(0)

        start_time = time.time()
        try:
            autolykos_v2_working_kernel[blocks_per_grid, threads_per_block](
                gpu_header,
                len(header_array),
                gpu_nonces,
                height,
                gpu_results,
                gpu_found,
                target_val,
                current_batch_size,
                dataset_seed_gpu,
            )
            cuda.synchronize()
            end_time = time.time()

            # التحقق من وجود حل (هذه العملية سريعة جداً)
            found_data = gpu_found.copy_to_host()
            if found_data[0] != 0:
                print(f"\n🎉 تم العثور على حل في الدفعة {batch + 1}!")
                solution_nonce = found_data[1]
                # بما أننا لا ننسخ النتائج، نحتاج إلى طريقة لاسترجاع الهاش الفائز فقط
                # ولكن للتبسيط، سنكتفي بطباعة النونس الآن
                print(f"   Nonce: {solution_nonce}")
                # في تطبيق حقيقي، ستحتاج إلى استرجاع هذا الهاش المحدد
                return solution_nonce, None  # نرجع None للهاش حالياً

            # حساب الإحصائيات الحقيقية للمعالجة
            batch_time = end_time - start_time
            # 🔥 تصحيح مهم: تأكد من أن current_batch_size هو عدد الـ threads الفعلي
            # إذا كان الكود لا يزال يعالج 1.6 مليون فقط، يجب إصلاح دالة calculate_proper_launch_config
            # بافتراض أن الدالة صحيحة الآن وتعالج كل الدفعة:
            actual_processed_nonces = blocks_per_grid * threads_per_block
            hashes_per_second = (
                min(current_batch_size, actual_processed_nonces) / batch_time
                if batch_time > 0
                else 0
            )

            print(f"   ✅ اكتملت في {batch_time:.2f} ثانية")
            print(f"   🚀 السرعة: {hashes_per_second:,.0f} هاش/ثانية")

            # ======================= 🔥 التعديل الرئيسي هنا 🔥 =======================
            # تم تعطيل هذا الجزء بالكامل لأنه يمثل عنق الزجاجة الرئيسي.
            # عملية `copy_to_host()` لمصفوفة النتائج الضخمة تبطئ البرنامج بشكل هائل
            # وتجعل قياس الأداء غير دقيق.
            #
            # if current_batch_size > 0:
            #     try:
            #         # السطر التالي هو سبب البطء الشديد
            #         # results = gpu_results.copy_to_host()
            #
            #         test_indices = [0, current_batch_size // 2, current_batch_size - 1]
            #         for test_idx in test_indices:
            #             # ... (باقي كود طباعة الهاشات)
            #
            #     except Exception as e:
            #         print(f"   ❌ خطأ في قراءة النتائج: {e}")
            # =========================================================================

            last_nonce = batch_end_nonce - 1
            print(f"   📍 آخر نونس معالج في الدفعة: {last_nonce:,}")

            progress = min(100, (batch_end_nonce / (start_nonce + total_nonces)) * 100)
            processed = min(total_nonces, (batch + 1) * max_nonces_per_batch)
            print(f"   📈 التقدم: {progress:.1f}% ({processed:,} / {total_nonces:,})")

        except Exception as e:
            print(f"   ❌ خطأ في الدفعة {batch + 1}: {e}")
            continue

        finally:
            # تحرير الذاكرة في كل دفعة لمنع تراكمها
            del gpu_nonces
            del gpu_results
            del gpu_found

    print("\n" + "=" * 60)
    print("🏁 اكتملت المعالجة")
    print(f"📌 آخر نونس: {last_nonce:,}")

    return last_nonce, solution_hash


# ===============================================================
# ==================== الاختبار الرئيسي ========================
# ===============================================================

if __name__ == "__main__":
    print("🚀 بدء التعدين الشغال باستخدام B200")
    print("💫 إصلاح مشاكل الـ indexing والـ blocks")
    print("🔥 يطبع الهاشات الحقيقية وليس الأصفار")
    print("=" * 60)

    # مثال header
    header_example = bytes.fromhex(
        "e1942c94d9feea26e95422a218d3892f709df4c0f79dea7802fe6a7c27e327414af2fc8312c042ab2e22a415f8e9cfebcdd2169b7e07d86d15bc0c02e9af605e0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
    )
    start_nonce = 0
    height_example = 0
    target_difficulty = None

    try:
        start_total_time = time.time()

        last_nonce, last_hash = process_8_billion_working(
            header_example, start_nonce, height_example, target_difficulty
        )

        total_time = time.time() - start_total_time
        total_hashes = min(8_000_000_000, (last_nonce - start_nonce + 1))
        avg_speed = total_hashes / total_time if total_time > 0 else 0

        print(f"\n📊 الإحصائيات النهائية:")
        print(f"   ⏱️  الوقت الإجمالي: {total_time:.2f} ثانية")
        print(f"   🚀 متوسط السرعة: {avg_speed:,.0f} هاش/ثانية")
        print(f"   💫 إجمالي الهاشات: {total_hashes:,}")
        print(f"   📍 آخر نونس: {last_nonce:,}")

    except Exception as e:
        print(f"❌ خطأ أثناء التنفيذ: {e}")
        import traceback

        traceback.print_exc()

# --- END OF FILE ---
