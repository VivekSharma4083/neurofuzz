/*
 * ============================================================================
 * NeuroFuzz Structured Benchmark Target - Phase 5
 * ============================================================================
 * WARNING: This file is an INTENTIONALLY VULNERABLE benchmark target created
 * solely for educational fuzzing research within the NeuroFuzz project.
 *
 * DO NOT USE OR DEPLOY THIS CODE IN ANY PRODUCTION ENVIRONMENT.
 *
 * Architecture:
 * - Multi-stage delimiter-based protocol parser ('NF' magic header)
 * - State machine with progressive authentication and privilege tiers
 * - Nested pipeline encoding / decoding stages (RAW, HEX, RLE)
 * - Deep sequential state progression (DEEP_STATE_1 through DEEP_STATE_MAX)
 * - Multiple deterministic, local, intentional bug/crash conditions:
 *     1. BUG_DIV_ZERO: Division by zero in math calculator
 *     2. BUG_INT_OVERFLOW: Integer truncation / overflow in write size
 *     3. BUG_OOB_READ: Out-of-bounds memory read in admin debug memory dump
 *     4. BUG_NULL_DEREF: Null pointer dereference in nested pipeline L3
 *     5. BUG_STACK_OVERFLOW: Unchecked stack copy in L5 RLE decoder
 *     6. BUG_DEEP_ABORT: Assertion abort in final DEEP_STATE_MAX handler
 *
 * Coverage:
 * - Emits deterministic __COV__:<IDENTIFIER> markers to stderr (unbuffered).
 * - Emits __COV__:DEPTH_<N> markers tracking maximum logical depth achieved.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

#define MAX_INPUT_SIZE 2048
#define COV_PREFIX "__COV__:"

static int g_persistent_mode = 0;
static int g_current_run_depth = 0;
static int g_current_run_cov_count = 0;

/* Explicit reset function to guarantee cross-iteration state isolation */
static void reset_target_state(void) {
    /* The benchmark parser allocates all session and token buffers on the stack
     * within process_structured_input(). No heap memory or static cross-iteration
     * buffers are maintained. This explicit reset function ensures any future global
     * flags, depth trackers, or counters are reset to pristine state prior to each input. */
    g_current_run_depth = 0;
    g_current_run_cov_count = 0;
}

/* Coverage emission helper */
static void emit_cov(const char *path_id) {
    g_current_run_cov_count++;
    if (g_persistent_mode) {
        fprintf(stdout, "%s%s\n", COV_PREFIX, path_id);
        fflush(stdout);
    } else {
        fprintf(stderr, "%s%s\n", COV_PREFIX, path_id);
        fflush(stderr);
    }
}

static void emit_depth(int depth) {
    if (depth > g_current_run_depth) {
        g_current_run_depth = depth;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "DEPTH_%02d", depth);
    emit_cov(buf);
}

/* Crash triggers */
static void trigger_segfault(void) {
    volatile int *null_ptr = NULL;
    *null_ptr = 0xDEAD;
}

/* Helper for safe string token parsing */
static int token_equals(const char *tok, const char *target) {
    return strcmp(tok, target) == 0;
}

static int token_starts_with(const char *tok, const char *prefix) {
    return strncmp(tok, prefix, strlen(prefix)) == 0;
}

/*
 * State Machine Enums
 */
typedef enum {
    PRIV_NONE = 0,
    PRIV_GUEST = 1,
    PRIV_DEV = 2,
    PRIV_ADMIN = 3,
    PRIV_ROOT = 4
} PrivilegeLevel;

typedef struct {
    int version;
    PrivilegeLevel priv;
    int deep_step;
    char username[32];
} ParserSession;

/* Reentrant portable delimiter splitter */
static int split_fields(char *input, char delimiter, char **out_fields, int max_fields) {
    int count = 0;
    char *p = input;
    out_fields[count++] = p;
    while (*p && count < max_fields) {
        if (*p == delimiter) {
            *p = '\0';
            p++;
            out_fields[count++] = p;
        } else {
            p++;
        }
    }
    return count;
}

/* Process the structured payload */
static void process_structured_input(const unsigned char *raw_data, size_t size) {
    emit_cov("PATH_ENTRY");
    emit_depth(1);

    if (size == 0) {
        emit_cov("PATH_EMPTY");
        return;
    }

    if (size < 2) {
        emit_cov("PATH_TOO_SHORT");
        return;
    }

    /* -------------------------------------------------------------
     * STAGE 1: Header Magic Check ('NF')
     * ------------------------------------------------------------- */
    if (raw_data[0] == 'N') {
        emit_cov("PATH_HDR_N");
        if (raw_data[1] == 'F') {
            emit_cov("PATH_HDR_VALID");
            emit_depth(2);
        } else {
            emit_cov("PATH_HDR_PARTIAL");
            return;
        }
    } else {
        emit_cov("PATH_HDR_INVALID");
        return;
    }

    if (size < 4) {
        emit_cov("PATH_VER_TOO_SHORT");
        return;
    }

    /* -------------------------------------------------------------
     * STAGE 2: Protocol Version ('01', '02', '03')
     * ------------------------------------------------------------- */
    ParserSession session;
    memset(&session, 0, sizeof(session));

    if (raw_data[2] == '0' && raw_data[3] == '1') {
        session.version = 1;
        emit_cov("PATH_VER_1");
        emit_depth(3);
    } else if (raw_data[2] == '0' && raw_data[3] == '2') {
        session.version = 2;
        emit_cov("PATH_VER_2");
        emit_depth(3);
    } else if (raw_data[2] == '0' && raw_data[3] == '3') {
        session.version = 3;
        emit_cov("PATH_VER_3");
        emit_depth(3);
    } else {
        emit_cov("PATH_VER_UNKNOWN");
        return;
    }

    /* Delimiter check after version */
    if (size <= 4) {
        emit_cov("PATH_TRUNCATED_AFTER_VER");
        return;
    }

    if (raw_data[4] != '|') {
        emit_cov("PATH_DELIM_MISSING");
        return;
    }
    emit_cov("PATH_DELIM_1_VALID");
    emit_depth(4);

    /* Copy input to a null-terminated work buffer for field tokenization */
    char text_buf[MAX_INPUT_SIZE + 1];
    size_t copy_len = size < MAX_INPUT_SIZE ? size : MAX_INPUT_SIZE;
    memcpy(text_buf, raw_data, copy_len);
    text_buf[copy_len] = '\0';

    /* Strip trailing newline / carriage returns for clean tokenization */
    while (copy_len > 0 && (text_buf[copy_len - 1] == '\n' || text_buf[copy_len - 1] == '\r')) {
        text_buf[--copy_len] = '\0';
    }

    /* Parse pipe-delimited fields */
    char *fields[16];
    int num_fields = split_fields(text_buf + 5, '|', fields, 16);

    if (num_fields == 0 || fields[0][0] == '\0') {
        emit_cov("PATH_NO_FIELDS");
        return;
    }
    emit_cov("PATH_FIELDS_PARSED");
    emit_depth(5);

    const char *cmd = fields[0];

    /* -------------------------------------------------------------
     * STAGE 3: Primary Command Dispatch
     * ------------------------------------------------------------- */
    if (token_equals(cmd, "PING")) {
        emit_cov("PATH_CMD_PING");
        emit_depth(6);
        if (num_fields > 1) {
            emit_cov("PATH_PING_PAYLOAD");
            if (strlen(fields[1]) > 10) {
                emit_cov("PATH_PING_LONG");
            }
        }
        return;
    }

    if (token_equals(cmd, "INFO")) {
        emit_cov("PATH_CMD_INFO");
        emit_depth(6);
        if (num_fields > 1 && token_equals(fields[1], "VERBOSE")) {
            emit_cov("PATH_INFO_VERBOSE");
        }
        return;
    }

    if (token_equals(cmd, "CALC")) {
        emit_cov("PATH_CMD_CALC");
        emit_depth(6);
        /* Format: NF01|CALC|OP=DIV|NUM=100|DEN=0 */
        int op_div = 0;
        long num = 100;
        long den = 1;

        for (int i = 1; i < num_fields; i++) {
            if (token_equals(fields[i], "OP=ADD")) {
                emit_cov("PATH_CALC_ADD");
            } else if (token_equals(fields[i], "OP=MUL")) {
                emit_cov("PATH_CALC_MUL");
            } else if (token_equals(fields[i], "OP=DIV")) {
                emit_cov("PATH_CALC_DIV");
                op_div = 1;
            } else if (token_starts_with(fields[i], "NUM=")) {
                num = strtol(fields[i] + 4, NULL, 10);
            } else if (token_starts_with(fields[i], "DEN=")) {
                den = strtol(fields[i] + 4, NULL, 10);
            }
        }

        if (op_div) {
            emit_cov("PATH_CALC_DIV_EXEC");
            if (den == 0) {
                /* BUG 1: Intentional Division by Zero */
                emit_cov("BUG_DIV_ZERO");
                volatile int zero = 0;
                volatile int res = (int)num / zero;
                (void)res;
            }
        }
        return;
    }

    if (token_equals(cmd, "WRITE")) {
        emit_cov("PATH_CMD_WRITE");
        emit_depth(6);
        /* Format: NF01|WRITE|SIZE=65535|DATA=xxxx */
        unsigned long user_size = 0;
        const char *data_ptr = NULL;

        for (int i = 1; i < num_fields; i++) {
            if (token_starts_with(fields[i], "SIZE=")) {
                user_size = strtoul(fields[i] + 5, NULL, 10);
            } else if (token_starts_with(fields[i], "DATA=")) {
                data_ptr = fields[i] + 5;
            }
        }

        if (user_size > 0 && data_ptr != NULL) {
            emit_cov("PATH_WRITE_HAS_DATA");
            /* BUG 2: Integer truncation bug */
            /* Truncating 32-bit unsigned long to 16-bit short creates negative/small alloc */
            short truncated_len = (short)user_size;
            if (user_size > 60000 && truncated_len < 0) {
                emit_cov("BUG_INT_OVERFLOW");
                char small_buf[16];
                /* Copy with original user size into small buffer -> crash */
                memcpy(small_buf, data_ptr, 64);
                trigger_segfault();
            }
        }
        return;
    }

    /* -------------------------------------------------------------
     * STAGE 4: Authentication State Machine
     * ------------------------------------------------------------- */
    if (token_equals(cmd, "AUTH")) {
        emit_cov("PATH_CMD_AUTH");
        emit_depth(6);

        char user_val[32] = {0};
        char token_val[64] = {0};

        for (int i = 1; i < num_fields; i++) {
            if (token_starts_with(fields[i], "USER=")) {
                snprintf(user_val, sizeof(user_val), "%s", fields[i] + 5);
                emit_cov("PATH_AUTH_HAS_USER");
            } else if (token_starts_with(fields[i], "TOKEN=") || token_starts_with(fields[i], "KEY=")) {
                const char *val_start = strchr(fields[i], '=') + 1;
                snprintf(token_val, sizeof(token_val), "%s", val_start);
                emit_cov("PATH_AUTH_HAS_TOKEN");
            }
        }

        /* User validation tier */
        if (token_equals(user_val, "guest")) {
            emit_cov("PATH_USER_GUEST");
            session.priv = PRIV_GUEST;
            emit_depth(7);
        } else if (token_equals(user_val, "dev")) {
            emit_cov("PATH_USER_DEV");
            if (strlen(token_val) >= 4) {
                session.priv = PRIV_DEV;
                emit_cov("PATH_AUTH_DEV_OK");
                emit_depth(8);
            } else {
                emit_cov("PATH_AUTH_DEV_FAIL");
            }
        } else if (token_equals(user_val, "admin")) {
            emit_cov("PATH_USER_ADMIN");
            if (token_starts_with(token_val, "adm_")) {
                session.priv = PRIV_ADMIN;
                emit_cov("PATH_AUTH_ADMIN_OK");
                emit_depth(9);
            } else {
                emit_cov("PATH_AUTH_ADMIN_FAIL");
            }
        } else if (token_equals(user_val, "root")) {
            emit_cov("PATH_USER_ROOT");
            if (token_equals(token_val, "!#ROOT#!")) {
                session.priv = PRIV_ROOT;
                emit_cov("PATH_AUTH_ROOT_OK");
                emit_depth(10);
            } else {
                emit_cov("PATH_AUTH_ROOT_FAIL");
            }
        } else {
            emit_cov("PATH_USER_UNKNOWN");
            return;
        }

        /* ---------------------------------------------------------
         * STAGE 5: Privileged Admin Operations
         * --------------------------------------------------------- */
        if (session.priv >= PRIV_ADMIN) {
            emit_cov("PATH_ADMIN_AUTHORIZED");
            emit_depth(11);

            for (int i = 1; i < num_fields; i++) {
                if (token_equals(fields[i], "MODE=NORMAL")) {
                    emit_cov("PATH_ADMIN_NORMAL");
                } else if (token_equals(fields[i], "MODE=MAINT")) {
                    emit_cov("PATH_ADMIN_MAINT");
                    emit_depth(12);
                } else if (token_equals(fields[i], "MODE=DEBUG")) {
                    emit_cov("PATH_ADMIN_DEBUG");
                    emit_depth(13);

                    /* Check debug parameters */
                    for (int j = 1; j < num_fields; j++) {
                        if (token_equals(fields[j], "DUMP=LOG")) {
                            emit_cov("PATH_DEBUG_DUMP_LOG");
                        } else if (token_starts_with(fields[j], "DUMP=MEM")) {
                            emit_cov("PATH_DEBUG_DUMP_MEM");
                            emit_depth(14);
                            /* Check for offset */
                            for (int k = 1; k < num_fields; k++) {
                                if (token_starts_with(fields[k], "OFF=")) {
                                    long offset = strtol(fields[k] + 4, NULL, 10);
                                    if (offset > 5000) {
                                        /* BUG 3: Intentional Out-Of-Bounds Read Crash */
                                        emit_cov("BUG_OOB_READ");
                                        volatile const char *ptr = (const char *)raw_data + offset * 1000;
                                        volatile char c = *ptr;
                                        (void)c;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    /* -------------------------------------------------------------
     * STAGE 6: Multi-Stage Deep Pipeline Parser ('DIAG')
     * ------------------------------------------------------------- */
    if (token_equals(cmd, "DIAG")) {
        emit_cov("PATH_CMD_DIAG");
        emit_depth(6);

        int nest_level = 0;
        int enc_type = 0; /* 1=RAW, 2=HEX, 3=RLE */
        const char *payload_str = NULL;

        for (int i = 1; i < num_fields; i++) {
            if (token_equals(fields[i], "NEST=L1")) {
                nest_level = 1;
                emit_cov("PATH_NEST_L1");
                emit_depth(7);
            } else if (token_equals(fields[i], "NEST=L2")) {
                nest_level = 2;
                emit_cov("PATH_NEST_L2");
                emit_depth(8);
            } else if (token_equals(fields[i], "NEST=L3")) {
                nest_level = 3;
                emit_cov("PATH_NEST_L3");
                emit_depth(9);
            } else if (token_equals(fields[i], "NEST=L4")) {
                nest_level = 4;
                emit_cov("PATH_NEST_L4");
                emit_depth(10);
            } else if (token_equals(fields[i], "NEST=L5")) {
                nest_level = 5;
                emit_cov("PATH_NEST_L5");
                emit_depth(11);
            } else if (token_equals(fields[i], "ENC=RAW")) {
                enc_type = 1;
                emit_cov("PATH_ENC_RAW");
            } else if (token_equals(fields[i], "ENC=HEX")) {
                enc_type = 2;
                emit_cov("PATH_ENC_HEX");
            } else if (token_equals(fields[i], "ENC=RLE")) {
                enc_type = 3;
                emit_cov("PATH_ENC_RLE");
            } else if (token_starts_with(fields[i], "PAY=")) {
                payload_str = fields[i] + 4;
                emit_cov("PATH_DIAG_HAS_PAYLOAD");
            }
        }

        /* BUG 4: Null Pointer Dereference in L3 with missing payload */
        if (nest_level == 3) {
            emit_cov("PATH_L3_ACTIVE");
            emit_depth(12);
            for (int i = 1; i < num_fields; i++) {
                if (token_equals(fields[i], "FLAG=TRIGGER_NULL")) {
                    emit_cov("BUG_NULL_DEREF");
                    trigger_segfault();
                }
            }
        }

        /* BUG 5: Stack Overflow in L5 RLE unpacker */
        if (nest_level == 5 && enc_type == 3 && payload_str != NULL) {
            emit_cov("PATH_L5_RLE_ACTIVE");
            emit_depth(13);
            /* If payload starts with 'R' and repeat count > 100 */
            if (payload_str[0] == 'R' && isdigit(payload_str[1])) {
                int repeat = atoi(payload_str + 1);
                emit_cov("PATH_RLE_PARSED");
                if (repeat > 100) {
                    emit_cov("BUG_STACK_OVERFLOW");
                    char small_stack[32];
                    memset(small_stack, 'A', repeat); /* Buffer overflow on stack */
                    trigger_segfault();
                }
            }
        }

        /* ---------------------------------------------------------
         * STAGE 7: Progressive State Machine Sequence
         * Must provide sequential STEP tokens to advance
         * --------------------------------------------------------- */
        int step = 0;
        for (int i = 1; i < num_fields; i++) {
            if (token_equals(fields[i], "STEP=1")) {
                step = 1;
                emit_cov("PATH_DEEP_STATE_1");
                emit_depth(14);
            } else if (step == 1 && token_equals(fields[i], "STEP=2")) {
                step = 2;
                emit_cov("PATH_DEEP_STATE_2");
                emit_depth(15);
            } else if (step == 2 && token_equals(fields[i], "STEP=3")) {
                step = 3;
                emit_cov("PATH_DEEP_STATE_3");
                emit_depth(16);
            } else if (step == 3 && token_equals(fields[i], "STEP=4")) {
                step = 4;
                emit_cov("PATH_DEEP_STATE_4");
                emit_depth(17);
            } else if (step == 4 && token_equals(fields[i], "STEP=5")) {
                step = 5;
                emit_cov("PATH_DEEP_STATE_5");
                emit_depth(18);
            }
        }

        if (step == 5) {
            emit_cov("PATH_DEEP_STATE_MAX");
            emit_depth(19);

            /* BUG 6: Deepest State Assertion Abort */
            for (int i = 1; i < num_fields; i++) {
                if (token_equals(fields[i], "EXEC=CRASH")) {
                    emit_cov("BUG_DEEP_ABORT");
                    abort();
                }
            }
        }
        return;
    }

    emit_cov("PATH_CMD_UNKNOWN");
}

static void run_persistent_mode(void) {
    g_persistent_mode = 1;
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    fprintf(stdout, "READY\n");
    fflush(stdout);

    char len_buf[64];
    while (fgets(len_buf, sizeof(len_buf), stdin)) {
        char *endp = NULL;
        long input_len = strtol(len_buf, &endp, 10);
        if (endp == len_buf) {
            if (len_buf[0] == '\0') break;
            continue;
        }
        if (input_len < 0) {
            break;
        }

        unsigned char buffer[MAX_INPUT_SIZE];
        size_t to_read = (size_t)input_len;
        size_t actual_read = 0;

        if (to_read > MAX_INPUT_SIZE) {
            actual_read = fread(buffer, 1, MAX_INPUT_SIZE, stdin);
            size_t remaining = to_read - actual_read;
            char drain[256];
            while (remaining > 0) {
                size_t chunk = remaining > sizeof(drain) ? sizeof(drain) : remaining;
                size_t drained = fread(drain, 1, chunk, stdin);
                if (drained == 0) {
                    break;
                }
                remaining -= drained;
            }
        } else if (to_read > 0) {
            actual_read = fread(buffer, 1, to_read, stdin);
        }

        reset_target_state();
        process_structured_input(buffer, actual_read);

        fprintf(stdout, "RESULT ok %d %d\n", g_current_run_cov_count, g_current_run_depth);
        fprintf(stdout, "READY\n");
        fflush(stdout);
    }
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    /* Suppress modal crash dialog on Windows */
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif

    if (argc > 1 && strcmp(argv[1], "--persistent") == 0) {
        run_persistent_mode();
        return 0;
    }

    unsigned char buffer[MAX_INPUT_SIZE];
    size_t bytes_read = 0;

    if (argc > 1) {
        FILE *fp = fopen(argv[1], "rb");
        if (!fp) {
            fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
            return 2;
        }
        bytes_read = fread(buffer, 1, sizeof(buffer), fp);
        fclose(fp);
    } else {
        bytes_read = fread(buffer, 1, sizeof(buffer), stdin);
    }

    reset_target_state();
    process_structured_input(buffer, bytes_read);
    return 0;
}

