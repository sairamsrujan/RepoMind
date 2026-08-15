# RepoMind — failure gallery

15 case(s), spread across failure categories. Each records the stage that failed, what the system returned, and the ground truth.

## 1. [fabricated_citation] What changed in the latest release?
- **Stage that failed:** reference validator  (source: metrics)
- **System returned:** (guard fail: fabricated_citations=1)
- **Ground truth:** (live query — no ground truth)

## 2. [incorrect_refusal] Why was a cache-size cap added to the repository?
- **Stage that failed:** guard / abstention  (source: eval)
- **System returned:** A cache-size cap was requested to be added to the repository to avoid unbounded memory growth [review_102_0]. However, there is no evidence within the provided chunks that the cache-size cap was actually added [none]. The existing evidence only shows the request for the cap in a review of Pull Request #102 [review_102_0], but it does not show any subsequent commit or pull request that implements this request within the indexed coverage window.
- **Ground truth:** To avoid unbounded memory growth.

## 3. [retrieval_miss] What change was made in issue #3469?
- **Stage that failed:** retrieval  (source: eval)
- **System returned:** In issue #3469, a pull request was made to fix the indentation on the quickstart documentation [pr_3469]. The indentation for two code blocks in the docs was corrected, as they were accidentally indented due to the RST conversion [pr_3469].
- **Ground truth:** The indentation on the quickstart documentation was fixed.

## 4. [incorrect_refusal] Why was an LRU cache added to the widget lookups?
- **Stage that failed:** guard / abstention  (source: eval)
- **System returned:** The LRU cache was added to the widget lookups because they were slow under heavy load [issue_2]. This issue was discussed and a solution was proposed in Pull Request #102, which added the caching layer [pr_102]. The cache was implemented to speed up repeated lookups [commit_deadbee20000]. 

Note that the initial implementation of the cache was requested to have a cache-size cap added to avoid unbounded memory growth [review_102_0], but there is no evidence provided within the given time frame th
- **Ground truth:** To speed up repeated lookups.

## 5. [retrieval_miss] Why were some commands marked as optional in the synopsis?
- **Stage that failed:** retrieval  (source: eval)
- **System returned:** Some commands were marked as optional in the synopsis [pr_3507] because the bracket syntax is a convention described in POSIX for optional arguments and operands [pr_3507]. This change was made to show optional subcommands in the help screen as `[COMMAND]` instead of `COMMAND` [pr_3507], and it closed issue #3059 [pr_3507]. The corresponding commit for this change is [commit_97210f24eb22].
- **Ground truth:** Some commands were marked as optional to indicate they are not required, addressing issue #3059.

## 6. [incorrect_refusal] Why was an LRU cache added to the widget lookup function?
- **Stage that failed:** guard / abstention  (source: eval)
- **System returned:** The LRU cache was added to the widget lookup function to speed up repeated lookups [commit_deadbee20000][pr_102]. This change was made in response to an issue reported where widget lookups were slow under heavy load [issue_2]. The addition of the cache was discussed and benchmarked in issue #2, which was closed by the pull request that introduced the cache [pr_102]. 

Note that the initial implementation of the cache did not include a cache-size cap, but a review of the pull request suggested ad
- **Ground truth:** To speed up repeated lookups under heavy load.

## 7. [retrieval_miss] Why was issue #3578 fixed?
- **Stage that failed:** retrieval  (source: eval)
- **System returned:** There is no evidence that issue #3578 was fixed [pr_3057][pr_3499][pr_3220][issue_3154][commit_0ab9d5d79bb8][issue_375]. In fact, issue #3578 is not mentioned at all in the provided evidence. Therefore, it is not possible to determine why issue #3578 was fixed, as there is no information about it. The evidence does not provide sufficient information to answer the question.
- **Ground truth:** The issue involved fixing double-bracketing of choices in the synopsis.

## 8. [incorrect_refusal] Why did the app start crashing on startup after the latest update?
- **Stage that failed:** guard / abstention  (source: eval)
- **System returned:** The question about the app crashing on startup after the latest update falls outside the provided evidence, as the evidence does not mention any update causing a crash, but rather fixes a crash that was already present [issue_1]. The latest update mentioned is Release v1.2.0, which fixes the startup crash and adds an LRU caching layer [release_v1.2.0]. The crash was fixed by null-checking the config before use [pr_101][commit_c0ffee100000]. 

Since the evidence does not provide information about
- **Ground truth:** The app was crashing because it had a null pointer exception when the config file was missing.

## 9. [retrieval_miss] Why does the `_less_uses_raw_mode` function need to inspect parsed tokens instead of a concatenated substring?
- **Stage that failed:** retrieval  (source: eval)
- **System returned:** The `_less_uses_raw_mode` function needs to inspect parsed tokens instead of a concatenated substring because the existing substring matching approach has limitations [issue_3416#2]. Specifically, it reports `color=False` when it should report `color=True` for certain cases [issue_3416#2]. The suggested fix involves inspecting parsed tokens from both sources to correctly determine if `less` uses raw mode [issue_3416#2]. This is necessary to avoid noisy output with literal `^[[31m...` sequences i
- **Ground truth:** The function needs to inspect parsed tokens because `less` invoked in default mode without `-R` or `-r` will treat ANSI escapes as literal sequences, making output noisier than no-color mode.

## 10. [incorrect_refusal] Why were some commands marked as optional in the synopsis?
- **Stage that failed:** guard / abstention  (source: eval)
- **System returned:** Some commands were marked as optional in the synopsis [pr_3507] because the bracket syntax is a convention described in POSIX for optional arguments and operands [pr_3507]. This change was made to show optional subcommands in the help screen as `[COMMAND]` instead of `COMMAND` [pr_3507], and it closed issue #3059 [pr_3507]. The corresponding commit for this change is [commit_97210f24eb22].
- **Ground truth:** Some commands were marked as optional to indicate they are not required, addressing issue #3059.

## 11. [retrieval_miss] What was fixed in the merge request #3390?
- **Stage that failed:** retrieval  (source: eval)
- **System returned:** The evidence for merge request #3390 does not provide a clear description of the change or the issue it addresses [pr_3390]. The comment in the pull request template was not replaced with a description of the change, and there is no other information provided about what was fixed [pr_3390]. Therefore, based on the available evidence, it is not possible to determine what was fixed in the merge request #3390.
- **Ground truth:** A bad merge was fixed.

## 12. [incorrect_refusal] Why was issue #3578 fixed?
- **Stage that failed:** guard / abstention  (source: eval)
- **System returned:** There is no evidence that issue #3578 was fixed [pr_3057][pr_3499][pr_3220][issue_3154][commit_0ab9d5d79bb8][issue_375]. In fact, issue #3578 is not mentioned at all in the provided evidence. Therefore, it is not possible to determine why issue #3578 was fixed, as there is no information about it. The evidence does not provide sufficient information to answer the question.
- **Ground truth:** The issue involved fixing double-bracketing of choices in the synopsis.

## 13. [retrieval_miss] Why was the `make_default_short_help` method marked as a private API?
- **Stage that failed:** retrieval  (source: eval)
- **System returned:** The `make_default_short_help` method was marked as a private API because it was explicitly stated as private API in a comment [pr_3250]. This change was enforced through a small pull request [pr_3250] that was approved by davidism and later merged into the codebase [commit_60cad4ad6583]. The reasoning behind this change is not explicitly stated in the provided evidence, but it is mentioned as part of a larger release [release_8.3.3#1] that included other changes to the API. The evidence does not
- **Ground truth:** It was likely done to indicate that this method is for internal use only and may change without notice, based on an issue comment.

## 14. [incorrect_refusal] Why are `stdin`, `stdout`, and `stderr` set to `None` or specific values in the `_get_handles` method?
- **Stage that failed:** guard / abstention  (source: eval)
- **System returned:** The `_get_handles` method sets `stdin`, `stdout`, and `stderr` to `None` or specific values based on the input parameters [issue_2412#4]. If `stdin` is `None`, it does nothing [issue_2412#4]. If `stdin` is `PIPE`, it creates a pipe using `os.pipe()` [issue_2412#4]. If `stdin` is `DEVNULL`, it sets `p2cread` to the result of `self._get_devnull()` [issue_2412#4]. If `stdin` is an integer, it sets `p2cread` to that integer [issue_2412#4]. If `stdin` is a file-like object, it sets `p2cread` to the r
- **Ground truth:** The `stdin`, `stdout`, and `stderr` parameters are checked against specific values like `PIPE` or `DEVNULL` to determine how file descriptors should be handled for subprocess communication.

## 15. [retrieval_miss] Why did get_parameter_source() start returning None inside ParamType.convert() and eager callbacks in version 8.4.0?
- **Stage that failed:** retrieval  (source: eval)
- **System returned:** In version 8.4.0, `get_parameter_source()` started returning `None` inside `ParamType.convert()` because the call to `ctx.set_parameter_source` was moved to after the arbitration step in `Parameter.handle_parse_result` [pr_3475#0]. This change was made in pull request #3404, which reworked `Parameter.handle_parse_result` [pr_3475#0]. As a result, a custom `ParamType.convert` that reads `ctx.get_parameter_source(param.name)` during conversion now sees `None` instead of the actual source [pr_3475#
- **Ground truth:** In version 8.4.0, set_parameter_source() was deferred until after type conversion and flag-group arbitration, so get_parameter_source() returned None inside ParamType.convert() and eager callbacks.
