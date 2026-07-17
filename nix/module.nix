{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.diskovod;
  settingsFormat = pkgs.formats.json { };
  configFile = settingsFormat.generate "diskovod.json" cfg.settings;
in
{
  options.services.diskovod = {
    enable = lib.mkEnableOption "Diskovod DM assistant";
    settings = lib.mkOption {
      inherit (settingsFormat) type;
      default = {
        host = "::1";
        port = 3090;
        public_url = "http://localhost:3090";
        log_level = "INFO";
      };
      description = ''
        Diskovod configuration rendered to a JSON file. Secrets must be
        supplied as file paths such as admin_password_file and secret_key_file.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.settings ? admin_password_file;
        message = "services.diskovod.settings.admin_password_file must be set";
      }
      {
        assertion = cfg.settings ? secret_key_file;
        message = "services.diskovod.settings.secret_key_file must be set";
      }
    ];

    systemd.services.diskovod = {
      description = "Diskovod DM assistant";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      restartTriggers = [ configFile ];
      serviceConfig = {
        DynamicUser = true;
        ExecSearchPath = lib.makeBinPath [ pkgs.diskovod ];
        ExecStart = "diskovod --config ${configFile}";
        StateDirectory = "diskovod";
        WorkingDirectory = "%S/diskovod";
        Restart = "on-failure";
        RestartSec = 5;
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
      };
    };
  };
}
