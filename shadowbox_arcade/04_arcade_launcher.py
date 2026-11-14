"""
=============================================================================
SHADOWBOX ARCADE — UNIFIED ARCADE SELECTION PORTAL (04_arcade_launcher.py)
Move. Strike. Slice.
=============================================================================
"""

import os
import sys
import time
import subprocess
import pygame

from player_profiles import get_profile_manager, sanitize_player_name
from music_manager import get_music_manager
from ui_audio import get_ui_audio

WINDOW_TITLE = "SHADOWBOX ARCADE — YOUR BODY IS THE CONTROLLER."
SCREEN_W = 1280
SCREEN_H = 720
FPS = 60

BOXER_SCRIPT = os.path.join(os.path.dirname(__file__), "02_shadow_boxer_game.py")
FRUIT_SCRIPT = os.path.join(os.path.dirname(__file__), "03_fruit_ninja_game.py")

class ArcadeLauncher:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        self.profile_mgr = get_profile_manager()
        active_player = self.profile_mgr.get_active_player()
        prefs = active_player.get("preferences", {})

        self.is_fullscreen = prefs.get("fullscreen", False)
        flags = pygame.DOUBLEBUF
        if self.is_fullscreen:
            flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        # Music & UI Audio Managers
        self.music = get_music_manager(prefs)
        self.ui_audio = get_ui_audio()
        self.last_hovered_card = 0

        # Fonts
        self.font_title = pygame.font.SysFont("impact", 68)
        self.font_tagline = pygame.font.SysFont("trebuchetms", 22, bold=True)
        self.font_card_title = pygame.font.SysFont("impact", 40)
        self.font_card_desc = pygame.font.SysFont("trebuchetms", 18)
        self.font_badge = pygame.font.SysFont("trebuchetms", 15, bold=True)
        self.font_btn = pygame.font.SysFont("impact", 26)
        self.font_ui = pygame.font.SysFont("trebuchetms", 20, bold=True)
        self.font_small = pygame.font.SysFont("trebuchetms", 16)

        # Card Geometries
        card_w, card_h = 490, 430
        y_pos = 175
        self.card_boxer = pygame.Rect(100, y_pos, card_w, card_h)
        self.card_fruit = pygame.Rect(690, y_pos, card_w, card_h)
        self.hovered_card = 0

        # UI Dialog States
        self.show_player_modal = False
        self.player_input_text = ""
        self.show_settings_modal = False

        # Leaderboard Context Menu & Deletion Dialog States
        self.context_menu_open = False
        self.context_menu_rect = None
        self.context_menu_player = None
        self.show_delete_confirm = False
        self.confirm_player = None
        self.player_row_rects = []
        self.delete_btn_rect = None
        self.cancel_btn_rect = None

        # Start menu background music
        self.music.play_track("menu", loop=-1, fade_ms=800)

    def launch_game(self, script_path):
        """Cleanly releases launcher resources, stops music, and executes chosen game."""
        active_player = self.profile_mgr.get_active_player()
        player_id = active_player.get("id", "guest")
        print(f"[Launcher] Launching {os.path.basename(script_path)} for player '{player_id}'...")

        # Play launch sting
        self.ui_audio.play("launch")

        # Fade out menu music
        self.music.stop(fade_ms=300)

        # Loading screen
        self.screen.fill((12, 14, 18))
        game_name = "SHADOW BOXING" if "boxer" in script_path else "FRUIT NINJA"
        msg = self.font_card_title.render(f"STARTING {game_name}...", True, (255, 215, 0))
        self.screen.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2 - 20))
        sub = self.font_tagline.render("INITIALIZING WEBCAM COMPUTER VISION PIPELINE", True, (180, 200, 220))
        self.screen.blit(sub, (SCREEN_W//2 - sub.get_width()//2, SCREEN_H//2 + 35))
        pygame.display.flip()

        # Release Pygame display and audio
        pygame.display.quit()
        time.sleep(0.3)

        # Verify target script exists
        if not os.path.isfile(script_path):
            print(f"[Launcher Error] Target script not found: {script_path}")
            self.ui_audio.play("error")
            return

        # Launch child process with player-id argument
        cmd = [sys.executable, script_path, "--player-id", player_id]
        if self.is_fullscreen:
            cmd.append("--fullscreen")

        exit_code = 0
        try:
            res = subprocess.run(cmd, check=False)
            exit_code = res.returncode
            if exit_code != 0:
                print(f"[Launcher Warning] {os.path.basename(script_path)} exited with code {exit_code}")
        except Exception as e:
            print(f"[Launcher Error] Failed to execute {script_path}: {e}")
            exit_code = -1

        # Re-initialize on return
        time.sleep(0.3)
        pygame.display.init()
        flags = pygame.DOUBLEBUF
        if self.is_fullscreen:
            flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
        pygame.display.set_caption(WINDOW_TITLE)

        # Refresh profiles and restore music
        self.profile_mgr.load()
        active_p = self.profile_mgr.get_active_player()
        self.music.play_track("menu", loop=-1, fade_ms=600)
        if exit_code != 0:
            self.ui_audio.play("error")
        print(f"[Launcher] Returned to Shadowbox Arcade. Active Player: {active_p['name']}")

    def run(self):
        running = True
        while running:
            active_player = self.profile_mgr.get_active_player()
            mouse_pos = pygame.mouse.get_pos()

            # Hover detection
            if not self.show_player_modal and not self.show_settings_modal:
                if self.card_boxer.collidepoint(mouse_pos):
                    self.hovered_card = 1
                elif self.card_fruit.collidepoint(mouse_pos):
                    self.hovered_card = 2
                else:
                    self.hovered_card = 0
            else:
                self.hovered_card = 0

            if self.hovered_card != self.last_hovered_card:
                if self.hovered_card != 0:
                    self.ui_audio.play("select")
                self.last_hovered_card = self.hovered_card

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.show_delete_confirm:
                        if event.key == pygame.K_ESCAPE:
                            self.ui_audio.play("close")
                            self.show_delete_confirm = False
                            self.confirm_player = None
                    elif self.context_menu_open:
                        if event.key == pygame.K_ESCAPE:
                            self.ui_audio.play("close")
                            self.context_menu_open = False
                            self.context_menu_player = None
                    elif self.show_player_modal:
                        if event.key == pygame.K_ESCAPE:
                            self.ui_audio.play("close")
                            self.show_player_modal = False
                            self.context_menu_open = False
                            self.context_menu_player = None
                        elif event.key == pygame.K_RETURN:
                            self.ui_audio.play("confirm")
                            if self.player_input_text.strip():
                                self.profile_mgr.create_player(self.player_input_text)
                                self.show_player_modal = False
                            self.player_input_text = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.player_input_text = self.player_input_text[:-1]
                        elif len(self.player_input_text) < 18 and event.unicode.isprintable():
                            self.player_input_text += event.unicode
                    elif self.show_settings_modal:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_s:
                            self.ui_audio.play("close")
                            self.show_settings_modal = False
                    else:
                        if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                            running = False
                        elif event.key == pygame.K_1:
                            self.launch_game(BOXER_SCRIPT)
                        elif event.key == pygame.K_2:
                            self.launch_game(FRUIT_SCRIPT)
                        elif event.key in [pygame.K_p, pygame.K_TAB, pygame.K_l]:
                            self.ui_audio.play("open")
                            self.show_player_modal = True
                            self.player_input_text = ""
                        elif event.key == pygame.K_s:
                            self.ui_audio.play("open")
                            self.show_settings_modal = True
                        elif event.key == pygame.K_m:
                            self.ui_audio.play("toggle")
                            muted = self.music.toggle_mute()
                            self.profile_mgr.update_preferences(active_player["id"], {"master_mute": muted})

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:  # RIGHT-CLICK: Custom Context Menu on Leaderboard Records
                        if self.show_player_modal and not self.show_delete_confirm and not self.show_settings_modal:
                            hit_record = False
                            for p_rect, p in self.player_row_rects:
                                if p_rect.collidepoint(mouse_pos):
                                    self.ui_audio.play("select")
                                    self.context_menu_player = p
                                    self.context_menu_open = True
                                    cm_w, cm_h = 175, 42
                                    cm_x = max(10, min(mouse_pos[0] + 4, SCREEN_W - cm_w - 10))
                                    cm_y = max(10, min(mouse_pos[1] + 4, SCREEN_H - cm_h - 10))
                                    self.context_menu_rect = pygame.Rect(cm_x, cm_y, cm_w, cm_h)
                                    hit_record = True
                                    break
                            if not hit_record:
                                self.context_menu_open = False
                                self.context_menu_player = None

                    elif event.button == 1:  # LEFT-CLICK
                        if self.show_delete_confirm:
                            self.handle_delete_confirm_click(mouse_pos)
                        elif self.context_menu_open:
                            if self.context_menu_rect and self.context_menu_rect.collidepoint(mouse_pos):
                                self.ui_audio.play("open")
                                self.context_menu_open = False
                                self.show_delete_confirm = True
                                self.confirm_player = self.context_menu_player
                            else:
                                self.context_menu_open = False
                                self.context_menu_player = None
                        elif self.show_player_modal:
                            self.handle_player_modal_click(mouse_pos)
                        elif self.show_settings_modal:
                            self.handle_settings_modal_click(mouse_pos)
                        else:
                            if self.hovered_card == 1:
                                self.launch_game(BOXER_SCRIPT)
                            elif self.hovered_card == 2:
                                self.launch_game(FRUIT_SCRIPT)
                            # Top-bar player badge click
                            badge_rect = pygame.Rect(30, 20, 320, 48)
                            if badge_rect.collidepoint(mouse_pos):
                                self.ui_audio.play("open")
                                self.show_player_modal = True
                                self.player_input_text = ""
                            # Top-bar leaderboard button click
                            lead_rect = pygame.Rect(SCREEN_W - 370, 20, 165, 48)
                            if lead_rect.collidepoint(mouse_pos):
                                self.ui_audio.play("open")
                                self.show_player_modal = True
                                self.player_input_text = ""
                            # Top-bar settings button click
                            set_rect = pygame.Rect(SCREEN_W - 190, 20, 160, 44)
                            if set_rect.collidepoint(mouse_pos):
                                self.ui_audio.play("open")
                                self.show_settings_modal = True

            # -----------------------------------------------------------------
            # RENDER
            # -----------------------------------------------------------------
            self.render_main_screen(active_player)

            if self.show_player_modal:
                self.render_player_modal()
            elif self.show_settings_modal:
                self.render_settings_modal(active_player)

            pygame.display.flip()
            self.clock.tick(FPS)

        self.music.cleanup()
        pygame.quit()

    # -------------------------------------------------------------------------
    # MAIN MENU SCREEN
    # -------------------------------------------------------------------------

    def render_main_screen(self, active_player):
        self.screen.fill((14, 16, 22))

        # 1. Top Bar: Player Profile & Quick Navigation
        p_name = active_player.get("name", "Guest")
        p_boxer_best = active_player.get("shadow_boxing", {}).get("high_score", 0)
        fn_data = active_player.get("fruit_ninja", {})
        p_fruit_best = max(fn_data.get("classic_high_score", 0), fn_data.get("arcade_high_score", 0), fn_data.get("zen_high_score", 0))

        # Player Badge Box (Clickable)
        badge_rect = pygame.Rect(30, 20, 320, 52)
        pygame.draw.rect(self.screen, (24, 28, 38), badge_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 130, 220), badge_rect, 2, border_radius=10)
        
        lbl_p = self.font_small.render("ACTIVE PLAYER [PRESS P / TAB]:", True, (150, 180, 220))
        self.screen.blit(lbl_p, (42, 25))
        name_t = self.font_ui.render(f"👤 {p_name.upper()}", True, (255, 255, 255))
        self.screen.blit(name_t, (42, 46))

        # Leaderboard Button (Top Center-Right)
        lead_rect = pygame.Rect(SCREEN_W - 375, 20, 170, 48)
        pygame.draw.rect(self.screen, (34, 30, 24), lead_rect, border_radius=8)
        pygame.draw.rect(self.screen, (220, 180, 50), lead_rect, 1, border_radius=8)
        lead_t = self.font_ui.render("🏆 RECORDS [L]", True, (255, 215, 60))
        self.screen.blit(lead_t, (lead_rect.centerx - lead_t.get_width()//2, lead_rect.centery - lead_t.get_height()//2))

        # Settings & Mute Status Buttons (Top Right)
        set_rect = pygame.Rect(SCREEN_W - 190, 20, 160, 48)
        pygame.draw.rect(self.screen, (28, 32, 44), set_rect, border_radius=8)
        pygame.draw.rect(self.screen, (90, 100, 130), set_rect, 1, border_radius=8)
        set_t = self.font_ui.render("⚙ SETTINGS [S]", True, (220, 225, 240))
        self.screen.blit(set_t, (set_rect.centerx - set_t.get_width()//2, set_rect.centery - set_t.get_height()//2))

        # 2. Main Title & Tagline
        title = self.font_title.render("SHADOWBOX ARCADE", True, (255, 215, 0))
        self.screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 38))
        
        tagline = self.font_tagline.render("YOUR BODY IS THE CONTROLLER.", True, (240, 90, 40))
        self.screen.blit(tagline, (SCREEN_W//2 - tagline.get_width()//2, 114))

        # 3. Game Card 1: SHADOW BOXING (Red/Orange Combat Theme)
        b_border = (255, 75, 45) if self.hovered_card == 1 else (75, 50, 50)
        b_bg = (32, 22, 24) if self.hovered_card == 1 else (22, 18, 20)
        pygame.draw.rect(self.screen, b_bg, self.card_boxer, border_radius=18)
        pygame.draw.rect(self.screen, b_border, self.card_boxer, 4 if self.hovered_card == 1 else 2, border_radius=18)

        # Card 1 Badge
        badge_b = self.font_badge.render("KINETIC AR COMBAT • POSE TRACKING", True, (255, 110, 80))
        self.screen.blit(badge_b, (self.card_boxer.x + 28, self.card_boxer.y + 24))

        # Card 1 Exact Name
        t1 = self.font_card_title.render("🥊 SHADOW BOXING", True, (255, 255, 255))
        self.screen.blit(t1, (self.card_boxer.x + 28, self.card_boxer.y + 48))

        # Card 1 Details & Player Personal Best
        boxer_info = [
            f"• Player Record: {p_boxer_best} pts",
            "• Natural Fist Punching & Biomechanical Extension",
            "• Acoustic 'KIAI' Battle Cry: 5x Critical Strike",
            "• Multimodal Sonic Clap Forcefield & Laser Parry",
            "• Armored Mecha Mini-Boss Waves & Health Bars",
            "• Dynamic Screen Shake & Procedural Heavy Punch DSP"
        ]
        by = self.card_boxer.y + 115
        for line in boxer_info:
            col = (255, 215, 0) if "Player Record" in line else (210, 210, 220)
            txt = self.font_card_desc.render(line, True, col)
            self.screen.blit(txt, (self.card_boxer.x + 28, by))
            by += 33

        # Button 1
        btn1 = pygame.Rect(self.card_boxer.x + 28, self.card_boxer.bottom - 70, self.card_boxer.w - 56, 48)
        btn1_col = (230, 55, 35) if self.hovered_card == 1 else (65, 35, 35)
        pygame.draw.rect(self.screen, btn1_col, btn1, border_radius=8)
        btn1_t = self.font_btn.render("PLAY SHADOW BOXING [PRESS 1]", True, (255, 255, 255))
        self.screen.blit(btn1_t, (btn1.centerx - btn1_t.get_width()//2, btn1.centery - btn1_t.get_height()//2))

        # 4. Game Card 2: FRUIT NINJA (Vibrant Emerald/Citrus Theme)
        f_border = (45, 225, 75) if self.hovered_card == 2 else (40, 65, 45)
        f_bg = (18, 30, 22) if self.hovered_card == 2 else (18, 22, 20)
        pygame.draw.rect(self.screen, f_bg, self.card_fruit, border_radius=18)
        pygame.draw.rect(self.screen, f_border, self.card_fruit, 4 if self.hovered_card == 2 else 2, border_radius=18)

        # Card 2 Badge
        badge_f = self.font_badge.render("HAND GESTURE SLICING • WEBCAM AR", True, (60, 230, 110))
        self.screen.blit(badge_f, (self.card_fruit.x + 28, self.card_fruit.y + 24))

        # Card 2 Exact Name
        t2 = self.font_card_title.render("🍉 FRUIT NINJA", True, (255, 255, 255))
        self.screen.blit(t2, (self.card_fruit.x + 28, self.card_fruit.y + 48))

        # Card 2 Details & Player Personal Best
        fruit_info = [
            f"• Player Record: {p_fruit_best} pts",
            "• Fist-Controlled Laser Sword (Sci-Fi Plasma)",
            "• Closed-Fist Grip Engine & Auto-Grip Mode",
            "• Continuous Swept Slicing (Zero Tunneling)",
            "• 3 Difficulties: Easy, Classic, and Expert",
            "• Interactive 5-Step Dojo Onboarding & Calibration"
        ]
        fy = self.card_fruit.y + 115
        for line in fruit_info:
            col = (255, 215, 0) if "Player Record" in line else (210, 210, 220)
            txt = self.font_card_desc.render(line, True, col)
            self.screen.blit(txt, (self.card_fruit.x + 28, fy))
            fy += 33

        # Button 2
        btn2 = pygame.Rect(self.card_fruit.x + 28, self.card_fruit.bottom - 70, self.card_fruit.w - 56, 48)
        btn2_col = (35, 195, 65) if self.hovered_card == 2 else (30, 60, 35)
        pygame.draw.rect(self.screen, btn2_col, btn2, border_radius=8)
        btn2_t = self.font_btn.render("PLAY FRUIT NINJA [PRESS 2]", True, (255, 255, 255))
        self.screen.blit(btn2_t, (btn2.centerx - btn2_t.get_width()//2, btn2.centery - btn2_t.get_height()//2))

        # 5. Bottom Control Footer
        footer = self.font_tagline.render("SELECT WITH MOUSE OR [1] SHADOW BOXING  •  [2] FRUIT NINJA  •  [L/P] LEADERBOARD  •  [S] SETTINGS  •  [ESC/Q] EXIT", True, (150, 165, 185))
        self.screen.blit(footer, (SCREEN_W//2 - footer.get_width()//2, SCREEN_H - 50))

    # -------------------------------------------------------------------------
    # ARCADE LEADERBOARD & PLAYER PROFILES MODAL (WITH RIGHT-CLICK DELETION)
    # -------------------------------------------------------------------------

    def render_player_modal(self):
        # Dim background
        dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        dim.fill((5, 8, 12, 225))
        self.screen.blit(dim, (0, 0))

        # Modal Box
        mw, mh = 720, 560
        mx = (SCREEN_W - mw) // 2
        my = (SCREEN_H - mh) // 2
        modal_rect = pygame.Rect(mx, my, mw, mh)
        pygame.draw.rect(self.screen, (20, 24, 34), modal_rect, border_radius=16)
        pygame.draw.rect(self.screen, (70, 130, 220), modal_rect, 2, border_radius=16)

        # Title Header
        title = self.font_card_title.render("ARCADE LEADERBOARD & PROFILES", True, (255, 215, 0))
        self.screen.blit(title, (mx + mw//2 - title.get_width()//2, my + 18))

        sub = self.font_small.render("Select active player  •  [Right-Click] any entry to Delete Record  •  [ESC] Close", True, (170, 190, 215))
        self.screen.blit(sub, (mx + mw//2 - sub.get_width()//2, my + 58))

        # Text Input Box & Add Button Row
        input_rect = pygame.Rect(mx + 45, my + 84, 380, 42)
        pygame.draw.rect(self.screen, (30, 36, 48), input_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 160, 255), input_rect, 2, border_radius=8)
        
        display_str = self.player_input_text if self.player_input_text else "Type name here (max 18 chars)..."
        text_col = (255, 255, 255) if self.player_input_text else (130, 145, 170)
        inp_t = self.font_ui.render(display_str, True, text_col)
        self.screen.blit(inp_t, (input_rect.x + 14, input_rect.centery - inp_t.get_height()//2))

        conf_rect = pygame.Rect(mx + 438, my + 84, 138, 42)
        pygame.draw.rect(self.screen, (40, 160, 70), conf_rect, border_radius=8)
        conf_t = self.font_ui.render("+ ADD NEW", True, (255, 255, 255))
        self.screen.blit(conf_t, (conf_rect.centerx - conf_t.get_width()//2, conf_rect.centery - conf_t.get_height()//2))

        guest_rect = pygame.Rect(mx + 588, my + 84, 88, 42)
        pygame.draw.rect(self.screen, (50, 60, 80), guest_rect, border_radius=8)
        guest_t = self.font_ui.render("GUEST", True, (220, 225, 240))
        self.screen.blit(guest_t, (guest_rect.centerx - guest_t.get_width()//2, guest_rect.centery - guest_t.get_height()//2))

        # Leaderboard Table Header Bar
        hdr_rect = pygame.Rect(mx + 45, my + 138, mw - 90, 28)
        pygame.draw.rect(self.screen, (28, 34, 46), hdr_rect, border_radius=6)
        
        h_rank = self.font_small.render("RANK", True, (150, 170, 195))
        h_name = self.font_small.render("PLAYER NAME", True, (150, 170, 195))
        h_box = self.font_small.render("BOXING", True, (255, 140, 100))
        h_fn = self.font_small.render("FRUIT", True, (100, 230, 140))
        h_best = self.font_small.render("RECORD", True, (255, 215, 60))
        h_stat = self.font_small.render("STATUS", True, (150, 170, 195))
        
        self.screen.blit(h_rank, (mx + 56, hdr_rect.centery - h_rank.get_height()//2))
        self.screen.blit(h_name, (mx + 115, hdr_rect.centery - h_name.get_height()//2))
        self.screen.blit(h_box, (mx + 315, hdr_rect.centery - h_box.get_height()//2))
        self.screen.blit(h_fn, (mx + 415, hdr_rect.centery - h_fn.get_height()//2))
        self.screen.blit(h_best, (mx + 495, hdr_rect.centery - h_best.get_height()//2))
        self.screen.blit(h_stat, (mx + 595, hdr_rect.centery - h_stat.get_height()//2))

        # Leaderboard Records List
        self.player_row_rects = []
        players = self.profile_mgr.list_players()
        active_player = self.profile_mgr.get_active_player()
        active_id = active_player.get("id")

        if not players:
            # Empty State
            empty_rect = pygame.Rect(mx + 45, my + 175, mw - 90, 310)
            pygame.draw.rect(self.screen, (16, 20, 28), empty_rect, border_radius=10)
            pygame.draw.rect(self.screen, (45, 55, 75), empty_rect, 1, border_radius=10)
            
            e_title = self.font_card_title.render("NO LEADERBOARD RECORDS YET", True, (210, 215, 230))
            self.screen.blit(e_title, (empty_rect.centerx - e_title.get_width()//2, empty_rect.centery - 35))
            
            e_sub = self.font_small.render("All records have been cleared. Enter a name above or play a game to set a score!", True, (140, 155, 180))
            self.screen.blit(e_sub, (empty_rect.centerx - e_sub.get_width()//2, empty_rect.centery + 12))
        else:
            mouse_pos = pygame.mouse.get_pos()
            py_offset = my + 172
            for idx, p in enumerate(players[:6], 1):
                p_rect = pygame.Rect(mx + 45, py_offset, mw - 90, 44)
                is_active = (p["id"] == active_id)
                is_hover = p_rect.collidepoint(mouse_pos) and not self.context_menu_open and not self.show_delete_confirm
                
                # Colors
                if is_active:
                    bg_col = (34, 46, 68) if is_hover else (26, 36, 54)
                    border_col = (80, 220, 130)
                    border_w = 2
                elif is_hover:
                    bg_col = (36, 44, 60)
                    border_col = (100, 160, 255)
                    border_w = 2
                else:
                    bg_col = (22, 26, 36)
                    border_col = (45, 52, 70)
                    border_w = 1
                    
                pygame.draw.rect(self.screen, bg_col, p_rect, border_radius=8)
                pygame.draw.rect(self.screen, border_col, p_rect, border_w, border_radius=8)
                
                # Rank
                if idx == 1:
                    r_col = (255, 215, 0)
                elif idx == 2:
                    r_col = (215, 220, 230)
                elif idx == 3:
                    r_col = (215, 140, 80)
                else:
                    r_col = (140, 155, 175)
                r_txt = self.font_small.render(f"#{idx}", True, r_col)
                self.screen.blit(r_txt, (mx + 58, p_rect.centery - r_txt.get_height()//2))

                # Name
                name_txt = self.font_ui.render(f"👤 {p['name']}", True, (255, 255, 255))
                self.screen.blit(name_txt, (mx + 115, p_rect.centery - name_txt.get_height()//2))

                # Boxing Score
                b_txt = self.font_small.render(f"{p['boxer_best']} pts", True, (255, 140, 100))
                self.screen.blit(b_txt, (mx + 315, p_rect.centery - b_txt.get_height()//2))

                # Fruit Ninja Score
                f_txt = self.font_small.render(f"{p['fruit_best']} pts", True, (100, 230, 140))
                self.screen.blit(f_txt, (mx + 415, p_rect.centery - f_txt.get_height()//2))

                # Combined High Score
                tot_txt = self.font_small.render(f"{p['high_score']} pts", True, (255, 215, 60))
                self.screen.blit(tot_txt, (mx + 495, p_rect.centery - tot_txt.get_height()//2))

                # Status tag
                if is_active:
                    st_txt = self.font_small.render("[ACTIVE]", True, (80, 255, 120))
                else:
                    st_txt = self.font_small.render("[SELECT]", True, (130, 150, 180))
                self.screen.blit(st_txt, (mx + 595, p_rect.centery - st_txt.get_height()//2))

                self.player_row_rects.append((p_rect, p))
                py_offset += 52

        # Close Hint
        hint = self.font_small.render("Right-Click on any row to Delete Record  •  Left-Click to Select  •  [ESC] Close", True, (150, 160, 180))
        self.screen.blit(hint, (mx + mw//2 - hint.get_width()//2, my + mh - 26))

        # ---------------------------------------------------------------------
        # CONTEXT MENU (RIGHT-CLICK)
        # ---------------------------------------------------------------------
        if self.context_menu_open and self.context_menu_player and self.context_menu_rect:
            cm = self.context_menu_rect
            # Drop shadow
            shadow_rect = cm.move(3, 3)
            s_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
            s_surf.fill((0, 0, 0, 180))
            self.screen.blit(s_surf, shadow_rect.topleft)

            mouse_pos = pygame.mouse.get_pos()
            is_cm_hover = cm.collidepoint(mouse_pos)
            cm_bg = (180, 35, 35) if is_cm_hover else (24, 28, 40)
            pygame.draw.rect(self.screen, cm_bg, cm, border_radius=6)
            pygame.draw.rect(self.screen, (235, 55, 55), cm, 2, border_radius=6)
            
            del_t = self.font_small.render("🗑 Delete Record", True, (255, 255, 255))
            self.screen.blit(del_t, (cm.centerx - del_t.get_width()//2, cm.centery - del_t.get_height()//2))

        # ---------------------------------------------------------------------
        # DELETE CONFIRMATION DIALOG MODAL
        # ---------------------------------------------------------------------
        if self.show_delete_confirm and self.confirm_player:
            c_dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            c_dim.fill((0, 0, 0, 215))
            self.screen.blit(c_dim, (0, 0))

            cw, ch = 560, 270
            cx = (SCREEN_W - cw) // 2
            cy = (SCREEN_H - ch) // 2
            confirm_box = pygame.Rect(cx, cy, cw, ch)
            pygame.draw.rect(self.screen, (22, 26, 36), confirm_box, border_radius=14)
            pygame.draw.rect(self.screen, (225, 50, 50), confirm_box, 2, border_radius=14)

            # Header
            t_conf = self.font_card_title.render("DELETE LEADERBOARD RECORD?", True, (255, 75, 75))
            self.screen.blit(t_conf, (cx + cw//2 - t_conf.get_width()//2, cy + 22))

            # Record Details
            p_name = self.confirm_player.get("name", "Player")
            b_sc = self.confirm_player.get("boxer_best", 0)
            f_sc = self.confirm_player.get("fruit_best", 0)
            p_id = self.confirm_player.get("id", "")

            name_line = self.font_ui.render(f"Player: 👤 {p_name}  (Record ID: {p_id})", True, (255, 255, 255))
            self.screen.blit(name_line, (cx + cw//2 - name_line.get_width()//2, cy + 80))

            score_line = self.font_small.render(f"High Scores: Shadow Boxing {b_sc} pts  •  Fruit Ninja {f_sc} pts", True, (255, 215, 60))
            self.screen.blit(score_line, (cx + cw//2 - score_line.get_width()//2, cy + 115))

            warn_line = self.font_small.render("This will permanently delete this record from persistent storage.", True, (215, 170, 170))
            self.screen.blit(warn_line, (cx + cw//2 - warn_line.get_width()//2, cy + 145))

            # Action Buttons
            mouse_pos = pygame.mouse.get_pos()
            self.cancel_btn_rect = pygame.Rect(cx + 45, cy + 195, 215, 46)
            self.delete_btn_rect = pygame.Rect(cx + 300, cy + 195, 215, 46)

            can_hover = self.cancel_btn_rect.collidepoint(mouse_pos)
            del_hover = self.delete_btn_rect.collidepoint(mouse_pos)

            pygame.draw.rect(self.screen, (60, 72, 95) if can_hover else (45, 52, 70), self.cancel_btn_rect, border_radius=8)
            pygame.draw.rect(self.screen, (100, 120, 150), self.cancel_btn_rect, 1, border_radius=8)
            btn_can_t = self.font_ui.render("CANCEL [ESC]", True, (225, 230, 245))
            self.screen.blit(btn_can_t, (self.cancel_btn_rect.centerx - btn_can_t.get_width()//2, self.cancel_btn_rect.centery - btn_can_t.get_height()//2))

            pygame.draw.rect(self.screen, (240, 45, 45) if del_hover else (200, 35, 35), self.delete_btn_rect, border_radius=8)
            pygame.draw.rect(self.screen, (255, 100, 100), self.delete_btn_rect, 1, border_radius=8)
            btn_del_t = self.font_ui.render("DELETE RECORD", True, (255, 255, 255))
            self.screen.blit(btn_del_t, (self.delete_btn_rect.centerx - btn_del_t.get_width()//2, self.delete_btn_rect.centery - btn_del_t.get_height()//2))

    def handle_delete_confirm_click(self, pos):
        """Processes clicks inside the delete confirmation modal."""
        if self.delete_btn_rect and self.delete_btn_rect.collidepoint(pos):
            if self.confirm_player:
                pid = self.confirm_player["id"]
                self.profile_mgr.delete_player(pid)
                self.ui_audio.play("confirm")
            self.show_delete_confirm = False
            self.confirm_player = None
            self.context_menu_open = False
            self.context_menu_player = None
        elif self.cancel_btn_rect and self.cancel_btn_rect.collidepoint(pos):
            self.ui_audio.play("close")
            self.show_delete_confirm = False
            self.confirm_player = None
            self.context_menu_open = False
            self.context_menu_player = None
        else:
            # Clicked outside confirmation box
            cw, ch = 560, 270
            cx = (SCREEN_W - cw) // 2
            cy = (SCREEN_H - ch) // 2
            confirm_box = pygame.Rect(cx, cy, cw, ch)
            if not confirm_box.collidepoint(pos):
                self.ui_audio.play("close")
                self.show_delete_confirm = False
                self.confirm_player = None
                self.context_menu_open = False
                self.context_menu_player = None

    def handle_player_modal_click(self, pos):
        mw, mh = 720, 560
        mx = (SCREEN_W - mw) // 2
        my = (SCREEN_H - mh) // 2
        
        conf_rect = pygame.Rect(mx + 438, my + 84, 138, 42)
        guest_rect = pygame.Rect(mx + 588, my + 84, 88, 42)

        if conf_rect.collidepoint(pos):
            self.ui_audio.play("confirm")
            if self.player_input_text.strip():
                self.profile_mgr.create_player(self.player_input_text)
                self.show_player_modal = False
            self.player_input_text = ""
        elif guest_rect.collidepoint(pos):
            self.ui_audio.play("confirm")
            self.profile_mgr.set_active_player("guest")
            self.show_player_modal = False
            self.player_input_text = ""
        else:
            # Check row clicks to select active player
            for p_rect, p in self.player_row_rects:
                if p_rect.collidepoint(pos):
                    self.ui_audio.play("confirm")
                    self.profile_mgr.set_active_player(p["id"])
                    self.show_player_modal = False
                    self.player_input_text = ""
                    break

    # -------------------------------------------------------------------------
    # SETTINGS & CUSTOMIZATION MODAL
    # -------------------------------------------------------------------------

    def render_settings_modal(self, active_player):
        dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        dim.fill((5, 8, 12, 225))
        self.screen.blit(dim, (0, 0))

        mw, mh = 720, 650
        mx = (SCREEN_W - mw) // 2
        my = (SCREEN_H - mh) // 2
        modal_rect = pygame.Rect(mx, my, mw, mh)
        pygame.draw.rect(self.screen, (22, 26, 36), modal_rect, border_radius=16)
        pygame.draw.rect(self.screen, (90, 110, 140), modal_rect, 2, border_radius=16)

        # Header
        title = self.font_card_title.render("ARCADE SETTINGS & PREFERENCES", True, (255, 215, 0))
        self.screen.blit(title, (mx + mw//2 - title.get_width()//2, my + 20))

        prefs = active_player.get("preferences", {})
        sy = my + 72

        # 1. Glove Size Setting (Shadow Boxing)
        g_size = prefs.get("glove_size", "Normal")
        if g_size == "Custom":
            g_size = "Normal"
        lbl1 = self.font_ui.render("Boxing Glove Size:", True, (255, 255, 255))
        self.screen.blit(lbl1, (mx + 45, sy))
        
        opts = [("Normal", "Normal (1.30x)"), ("Large", "Large (1.45x)")]
        ox = mx + 265
        for opt_val, opt_lbl in opts:
            r = pygame.Rect(ox, sy - 4, 185, 30)
            sel = (g_size == opt_val) or (opt_val == "Normal" and g_size not in ("Normal", "Large"))
            pygame.draw.rect(self.screen, (40, 130, 240) if sel else (35, 40, 55), r, border_radius=6)
            t = self.font_small.render(opt_lbl, True, (255, 255, 255) if sel else (180, 190, 200))
            self.screen.blit(t, (r.centerx - t.get_width()//2, r.centery - t.get_height()//2))
            ox += 195

        # 2. Glove Cosmetic Skin (Shadow Boxing)
        sy += 48
        g_skin = prefs.get("glove_skin", "Classic Red")
        lbl2 = self.font_ui.render("Boxing Glove Skin:", True, (255, 255, 255))
        self.screen.blit(lbl2, (mx + 45, sy))

        skins = ["Classic Red", "Electric Blue", "Neon Purple", "Golden"]
        ox = mx + 235
        for skin in skins:
            r = pygame.Rect(ox, sy - 4, 105, 30)
            sel = (g_skin == skin)
            pygame.draw.rect(self.screen, (220, 80, 40) if sel else (35, 40, 55), r, border_radius=6)
            t = self.font_small.render(skin.split()[0], True, (255, 255, 255) if sel else (180, 190, 200))
            self.screen.blit(t, (r.centerx - t.get_width()//2, r.centery - t.get_height()//2))
            ox += 112

        # 3. Fruit Ninja Difficulty Setting
        sy += 52
        fn_diff = prefs.get("difficulty", "EASY")
        lbl_diff = self.font_ui.render("Fruit Ninja Difficulty:", True, (255, 255, 255))
        self.screen.blit(lbl_diff, (mx + 45, sy))

        diff_opts = [("EASY", (60, 220, 100)), ("CLASSIC", (255, 215, 60)), ("EXPERT", (255, 80, 80))]
        ox = mx + 265
        for d_name, d_col in diff_opts:
            r = pygame.Rect(ox, sy - 4, 120, 30)
            sel = (fn_diff == d_name)
            pygame.draw.rect(self.screen, d_col if sel else (35, 40, 55), r, border_radius=6)
            t = self.font_small.render(d_name, True, (20, 24, 30) if sel else (180, 190, 200))
            self.screen.blit(t, (r.centerx - t.get_width()//2, r.centery - t.get_height()//2))
            ox += 128

        # 4. Fruit Ninja Auto-Grip Setting
        sy += 48
        fn_ag = prefs.get("auto_grip", False)
        lbl_ag = self.font_ui.render("Laser Sword Auto-Grip:", True, (255, 255, 255))
        self.screen.blit(lbl_ag, (mx + 45, sy))

        ag_opts = [(False, "OFF (Fist Clench)"), (True, "ON (Always Blade)")]
        ox = mx + 265
        for ag_val, ag_lbl in ag_opts:
            r = pygame.Rect(ox, sy - 4, 185, 30)
            sel = (fn_ag == ag_val)
            pygame.draw.rect(self.screen, (40, 180, 120) if sel else (35, 40, 55), r, border_radius=6)
            t = self.font_small.render(ag_lbl, True, (255, 255, 255) if sel else (180, 190, 200))
            self.screen.blit(t, (r.centerx - t.get_width()//2, r.centery - t.get_height()//2))
            ox += 195

        # 5. Audio: Music Volume
        sy += 52
        m_vol = int(self.music.music_volume * 100)
        lbl3 = self.font_ui.render(f"Music Volume: {m_vol}%", True, (255, 255, 255))
        self.screen.blit(lbl3, (mx + 45, sy))

        r_m_minus = pygame.Rect(mx + 300, sy - 4, 40, 30)
        r_m_plus = pygame.Rect(mx + 355, sy - 4, 40, 30)
        pygame.draw.rect(self.screen, (50, 60, 80), r_m_minus, border_radius=6)
        pygame.draw.rect(self.screen, (50, 60, 80), r_m_plus, border_radius=6)
        self.screen.blit(self.font_ui.render("-", True, (255, 255, 255)), (r_m_minus.centerx - 5, r_m_minus.centery - 12))
        self.screen.blit(self.font_ui.render("+", True, (255, 255, 255)), (r_m_plus.centerx - 6, r_m_plus.centery - 12))

        # 6. Audio: SFX Volume
        sy += 46
        s_vol = int(self.music.sfx_volume * 100)
        lbl4 = self.font_ui.render(f"SFX Volume: {s_vol}%", True, (255, 255, 255))
        self.screen.blit(lbl4, (mx + 45, sy))

        r_s_minus = pygame.Rect(mx + 300, sy - 4, 40, 30)
        r_s_plus = pygame.Rect(mx + 355, sy - 4, 40, 30)
        pygame.draw.rect(self.screen, (50, 60, 80), r_s_minus, border_radius=6)
        pygame.draw.rect(self.screen, (50, 60, 80), r_s_plus, border_radius=6)
        self.screen.blit(self.font_ui.render("-", True, (255, 255, 255)), (r_s_minus.centerx - 5, r_s_minus.centery - 12))
        self.screen.blit(self.font_ui.render("+", True, (255, 255, 255)), (r_s_plus.centerx - 6, r_s_plus.centery - 12))

        # 7. Kiai Microphone Sensitivity
        sy += 46
        m_sens = prefs.get("mic_sensitivity", 1.0)
        lbl_mic = self.font_ui.render("Kiai Mic Sensitivity:", True, (255, 255, 255))
        self.screen.blit(lbl_mic, (mx + 45, sy))

        mic_opts = [(0.7, "Low (0.7x)"), (1.0, "Norm (1.0x)"), (1.4, "High (1.4x)")]
        ox = mx + 265
        for s_val, s_lbl in mic_opts:
            r = pygame.Rect(ox, sy - 4, 120, 30)
            sel = abs(m_sens - s_val) < 0.15
            pygame.draw.rect(self.screen, (0, 200, 160) if sel else (35, 40, 55), r, border_radius=6)
            t = self.font_small.render(s_lbl, True, (20, 24, 30) if sel else (180, 190, 200))
            self.screen.blit(t, (r.centerx - t.get_width()//2, r.centery - t.get_height()//2))
            ox += 128

        # 8. Master Mute & Fullscreen Toggles
        sy += 48
        mute_t = "MUTED" if self.music.is_muted else "AUDIO ACTIVE"
        mute_col = (255, 75, 75) if self.music.is_muted else (75, 220, 100)
        r_mute = pygame.Rect(mx + 45, sy, 220, 34)
        pygame.draw.rect(self.screen, (35, 40, 55), r_mute, border_radius=6)
        t_mute = self.font_small.render(f"MUTE: {mute_t}", True, mute_col)
        self.screen.blit(t_mute, (r_mute.centerx - t_mute.get_width()//2, r_mute.centery - t_mute.get_height()//2))

        r_fs = pygame.Rect(mx + 290, sy, 220, 34)
        fs_str = "FULLSCREEN: ON" if self.is_fullscreen else "FULLSCREEN: OFF"
        pygame.draw.rect(self.screen, (35, 40, 55), r_fs, border_radius=6)
        t_fs = self.font_small.render(fs_str, True, (220, 225, 240))
        self.screen.blit(t_fs, (r_fs.centerx - t_fs.get_width()//2, r_fs.centery - t_fs.get_height()//2))

        # Close Button
        sy += 55
        r_close = pygame.Rect(mx + mw//2 - 90, sy, 180, 40)
        pygame.draw.rect(self.screen, (40, 140, 220), r_close, border_radius=8)
        c_t = self.font_ui.render("DONE [ESC]", True, (255, 255, 255))
        self.screen.blit(c_t, (r_close.centerx - c_t.get_width()//2, r_close.centery - c_t.get_height()//2))

    def handle_settings_modal_click(self, pos):
        mw, mh = 720, 650
        mx = (SCREEN_W - mw) // 2
        my = (SCREEN_H - mh) // 2
        active_p = self.profile_mgr.get_active_player()
        pid = active_p["id"]
        prefs = active_p.get("preferences", {})

        # 1. Glove Size options
        sy1 = my + 72
        opts = ["Normal", "Large"]
        ox = mx + 265
        for opt_val in opts:
            r = pygame.Rect(ox, sy1 - 4, 185, 30)
            if r.collidepoint(pos):
                self.ui_audio.play("toggle")
                self.profile_mgr.update_preferences(pid, {"glove_size": opt_val})
                return
            ox += 195

        # 2. Glove Skin options
        sy2 = sy1 + 48
        skins = ["Classic Red", "Electric Blue", "Neon Purple", "Golden"]
        ox = mx + 235
        for skin in skins:
            r = pygame.Rect(ox, sy2 - 4, 105, 30)
            if r.collidepoint(pos):
                self.ui_audio.play("toggle")
                self.profile_mgr.update_preferences(pid, {"glove_skin": skin})
                return
            ox += 112

        # 3. Fruit Ninja Difficulty options
        sy_diff = sy2 + 52
        diff_opts = ["EASY", "CLASSIC", "EXPERT"]
        ox = mx + 265
        for d_name in diff_opts:
            r = pygame.Rect(ox, sy_diff - 4, 120, 30)
            if r.collidepoint(pos):
                self.ui_audio.play("toggle")
                self.profile_mgr.update_preferences(pid, {"difficulty": d_name})
                return
            ox += 128

        # 4. Fruit Ninja Auto-Grip options
        sy_ag = sy_diff + 48
        ag_opts = [False, True]
        ox = mx + 265
        for ag_val in ag_opts:
            r = pygame.Rect(ox, sy_ag - 4, 185, 30)
            if r.collidepoint(pos):
                self.ui_audio.play("toggle")
                self.profile_mgr.update_preferences(pid, {"auto_grip": ag_val})
                return
            ox += 195

        # 5. Music Volume
        sy3 = sy_ag + 52
        r_m_minus = pygame.Rect(mx + 300, sy3 - 4, 40, 30)
        r_m_plus = pygame.Rect(mx + 355, sy3 - 4, 40, 30)
        if r_m_minus.collidepoint(pos):
            self.ui_audio.play("click")
            new_v = max(0.0, self.music.music_volume - 0.1)
            self.music.set_music_volume(new_v)
            self.profile_mgr.update_preferences(pid, {"music_volume": new_v})
            return
        elif r_m_plus.collidepoint(pos):
            self.ui_audio.play("click")
            new_v = min(1.0, self.music.music_volume + 0.1)
            self.music.set_music_volume(new_v)
            self.profile_mgr.update_preferences(pid, {"music_volume": new_v})
            return

        # 6. SFX Volume
        sy4 = sy3 + 46
        r_s_minus = pygame.Rect(mx + 300, sy4 - 4, 40, 30)
        r_s_plus = pygame.Rect(mx + 355, sy4 - 4, 40, 30)
        if r_s_minus.collidepoint(pos):
            self.ui_audio.play("click")
            new_v = max(0.0, self.music.sfx_volume - 0.1)
            self.music.set_sfx_volume(new_v)
            self.profile_mgr.update_preferences(pid, {"sfx_volume": new_v})
            return
        elif r_s_plus.collidepoint(pos):
            self.ui_audio.play("click")
            new_v = min(1.0, self.music.sfx_volume + 0.1)
            self.music.set_sfx_volume(new_v)
            self.profile_mgr.update_preferences(pid, {"sfx_volume": new_v})
            return

        # 7. Mic Sensitivity
        sy_mic = sy4 + 46
        mic_opts = [0.7, 1.0, 1.4]
        ox = mx + 265
        for s_val in mic_opts:
            r = pygame.Rect(ox, sy_mic - 4, 120, 30)
            if r.collidepoint(pos):
                self.ui_audio.play("toggle")
                self.profile_mgr.update_preferences(pid, {"mic_sensitivity": s_val})
                return
            ox += 128

        # 8. Mute toggle
        sy5 = sy_mic + 48
        r_mute = pygame.Rect(mx + 45, sy5, 220, 34)
        if r_mute.collidepoint(pos):
            self.ui_audio.play("toggle")
            muted = self.music.toggle_mute()
            self.profile_mgr.update_preferences(pid, {"master_mute": muted})
            return

        # Fullscreen toggle
        r_fs = pygame.Rect(mx + 290, sy5, 220, 34)
        if r_fs.collidepoint(pos):
            self.ui_audio.play("toggle")
            self.is_fullscreen = not self.is_fullscreen
            self.profile_mgr.update_preferences(pid, {"fullscreen": self.is_fullscreen})
            pygame.display.quit()
            pygame.display.init()
            flags = pygame.DOUBLEBUF
            if self.is_fullscreen:
                flags |= pygame.FULLSCREEN
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
            return

        # Close
        sy6 = sy5 + 55
        r_close = pygame.Rect(mx + mw//2 - 90, sy6, 180, 40)
        if r_close.collidepoint(pos):
            self.ui_audio.play("close")
            self.show_settings_modal = False

if __name__ == "__main__":
    launcher = ArcadeLauncher()
    launcher.run()
