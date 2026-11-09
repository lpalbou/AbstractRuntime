### Fixed

- **The hard stop signals only this home's verified loop and says killed only on a confirmed exit**
  (`identity/life.py` `hard_stop_loop`; Codex review 2026-11-09 #4, #7). Before any signal the incarnation is
  verified - alive (not unknown), not the caller, the recorded start-time token equal to the OS's - and the process
  must be THIS home's loop invocation: `<python> -m abstractruntime.identity.life --home <this home>`, the module
  right after the interpreter (the module name as a later argument, another home's loop, a missing token, an
  unreadable start time or command line all refuse: fail closed). Before escalating to SIGKILL the incarnation is
  verified again. `killed` is true only on a confirmed exit (reaped, ESRCH, or a zombie); a denied permission
  (EPERM) or an unreadable status is "unknown", never a death, and gets no SIGKILL.
