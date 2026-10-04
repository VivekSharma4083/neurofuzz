/*
 * ============================================================================
 * NeuroFuzz Educational Target Program - Phase 2 (Coverage Instrumented)
 * ============================================================================
 * WARNING: This file is an INTENTIONALLY VULNERABLE toy target created solely
 * for educational fuzz testing purposes within the NeuroFuzz research project.
 *
 * DO NOT USE OR DEPLOY THIS CODE IN ANY PRODUCTION ENVIRONMENT.
 *
 * IMPORTANT NOTE ON INSTRUMENTATION:
 * This Phase 2 implementation uses educational path/branch instrumentation via
 * `emit_cov()`. It exposes stable logical coverage identifiers to stderr
 * (e.g., `__COV__:PATH_HDR_FUZZ`). This educational instrumentation allows
 * demonstrating coverage-guided fuzzing, new frontier discovery, and corpus
 * admission without requiring complex compiler plugins. Production-grade
 * edge coverage (such as AFL++ bitmap instrumentation or SanitizerCoverage)
 * will be introduced in later phases.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define MAX_INPUT_SIZE 1024
#define COV_PREFIX "__COV__:"

/*
 * Educational coverage emitter.
 * Emits stable logical execution path identifiers to stderr.
 * Note: stderr is flushed immediately so coverage prior to a crash or
 * abnormal abort is preserved.
 */
static void emit_cov(const char *path_id) {
    fprintf(stderr, "%s%s\n", COV_PREFIX, path_id);
    fflush(stderr);
}

/* Helper to trigger a memory access violation (SIGSEGV / STATUS_ACCESS_VIOLATION) */
static void trigger_segfault(void) {
    volatile int *null_ptr = NULL;
    *null_ptr = 0xDEAD;
}

/* Parse and process the input buffer with coverage instrumentation */
static void process_input(const unsigned char *data, size_t size) {
    emit_cov("PATH_ENTRY");

    if (size == 0) {
        emit_cov("PATH_EMPTY");
        printf("[Target] Empty input received.\n");
        return;
    }

    /* Tier 1: Shallow bug (triggered easily by byte flip / replacement) */
    if (data[0] == '!' || data[0] == 0xFF) {
        emit_cov("PATH_SHALLOW_CRASH");
        printf("[Target] Triggering Tier 1 crash on special character (0x%02X)!\n", data[0]);
        trigger_segfault();
        return;
    }

    if (size < 4) {
        emit_cov("PATH_TOO_SHORT");
        printf("[Target] Input too short (%lu bytes).\n", (unsigned long)size);
        return;
    }

    /* Tier 2: Magic header inspection */

    /* Header: CRSH */
    if (data[0] == 'C' && data[1] == 'R' && data[2] == 'S' && data[3] == 'H') {
        emit_cov("PATH_HDR_CRSH");
        printf("[Target] Header CRSH matched! Triggering segfault...\n");
        trigger_segfault();
        return;
    }

    /* Header: NEUR */
    if (data[0] == 'N' && data[1] == 'E' && data[2] == 'U' && data[3] == 'R') {
        emit_cov("PATH_HDR_NEUR");
        printf("[Target] Header matched: NEUR\n");
        if (size >= 5 && data[4] == 'D') {
            emit_cov("PATH_NEUR_CMD_D");
            printf("[Target] NEUR 'D' command crash!\n");
            trigger_segfault();
        } else {
            emit_cov("PATH_NEUR_CMD_OTHER");
        }
        return;
    }

    /* Header: TEST */
    if (data[0] == 'T' && data[1] == 'E' && data[2] == 'S' && data[3] == 'T') {
        emit_cov("PATH_HDR_TEST");
        printf("[Target] Header matched: TEST\n");
        if (size >= 5 && data[4] == 'K') {
            emit_cov("PATH_TEST_CMD_K");
            printf("[Target] TEST 'K' crash!\n");
            abort();
        } else {
            emit_cov("PATH_TEST_CMD_OTHER");
        }
        return;
    }

    /* Header: FUZZ */
    if (data[0] == 'F' && data[1] == 'U' && data[2] == 'Z' && data[3] == 'Z') {
        emit_cov("PATH_HDR_FUZZ");
        printf("[Target] Header matched: FUZZ\n");

        /* Length check bug: payload over 16 bytes causes buffer overflow */
        if (size > 16) {
            emit_cov("PATH_FUZZ_OVERSIZED");
            printf("[Target] Payload exceeds 16 bytes (%lu bytes)! Buffer overflow!\n", (unsigned long)size);
            trigger_segfault();
            return;
        }

        if (size < 5) {
            emit_cov("PATH_FUZZ_NO_CMD");
            printf("[Target] Missing command byte after FUZZ.\n");
            return;
        }

        unsigned char cmd = data[4];
        switch (cmd) {
            case 'E': /* Normal echo */
                emit_cov("PATH_FUZZ_CMD_ECHO");
                printf("[Target] Echo command: %.*s\n", (int)(size - 5), data + 5);
                break;

            case 'C': /* Crash via abort */
            case '!':
            case 'X':
            case 0:
                emit_cov("PATH_FUZZ_CMD_ABORT");
                printf("[Target] Abort crash triggered by command '%c' (0x%02X)!\n", cmd, cmd);
                abort();
                break;

            case 'Z': /* Crash via division by zero */
            case '0':
                emit_cov("PATH_FUZZ_CMD_DIVZERO");
                printf("[Target] Division by zero triggered by command '%c'!\n", cmd);
                {
                    volatile int zero = 0;
                    volatile int res = 100 / zero;
                    (void)res;
                }
                break;

            case 'O': /* Crash via stack buffer overflow */
                emit_cov("PATH_FUZZ_CMD_OVERFLOW");
                printf("[Target] Stack overflow copy triggered by command 'O'!\n");
                {
                    volatile char small_stack_buf[8];
                    for (size_t i = 0; i < size; i++) {
                        small_stack_buf[i] = (char)data[i];
                    }
                    (void)small_stack_buf[0];
                    trigger_segfault();
                }
                break;

            case 'H': /* Hang / infinite loop (evaluates timeout detection) */
                emit_cov("PATH_FUZZ_CMD_HANG");
                printf("[Target] Hang command 'H' received! Spinning in infinite loop...\n");
                while (1) {
                    /* Spin until killed by executor timeout */
                }
                break;

            default:
                emit_cov("PATH_FUZZ_CMD_UNKNOWN");
                printf("[Target] Unhandled FUZZ command: '%c' (0x%02X)\n", cmd, cmd);
                break;
        }
        return;
    }

    emit_cov("PATH_HDR_UNKNOWN");
    printf("[Target] Unrecognized header (0x%02X%02X%02X%02X)\n",
           data[0], data[1], data[2], data[3]);
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    /* Suppress Windows modal crash dialog so exceptions exit immediately with their NTSTATUS */
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif

    unsigned char buffer[MAX_INPUT_SIZE];
    size_t bytes_read = 0;

    if (argc > 1) {
        /* Read from specified file */
        FILE *fp = fopen(argv[1], "rb");
        if (!fp) {
            fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
            return 2;
        }
        bytes_read = fread(buffer, 1, sizeof(buffer), fp);
        fclose(fp);
    } else {
        /* Read from stdin */
        bytes_read = fread(buffer, 1, sizeof(buffer), stdin);
    }

    process_input(buffer, bytes_read);
    return 0;
}
