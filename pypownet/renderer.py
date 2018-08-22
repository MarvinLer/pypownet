__author__ = 'marvinler'
import pygame
import math
from pygame import gfxdraw
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pylab

case_layouts = {
    14: [(0.0, 210.0), (150.0, 0.0), (540.0, 0.0), (540.0, 240.0), (180.0, 240.0), (180.0, 360.0), (540.0, 300.0),
         (600.0, 300.0), (540.0, 360.0), (420.0, 420.0), (300.0, 480.0), (90.0, 600.0), (180.0, 600.0),
         (420.0, 540.0)],

    30: [(-452.0, -423.0), (-315.0, -558.0), (-318.0, -430.0), (-187.0, -428.0), (44.0, -561.0), (112.0, -446.0),
         (106.0, -509.0), (294.0, -452.0), (88.0, -315.0), (128.0, -285.0), (-29.0, -312.0), (-183.0, -267.0),
         (-312.0, -269.0), (-285.0, -185.0), (-185.0, -75.0), (-134.0, -203.0), (-14.0, -175.0), (-27.0, -52.0),
         (172.0, -36.0), (130.0, -79.0), (168.0, -133.0), (214.0, -160.0), (-17.0, 20.0), (221.0, 21.0),
         (108.0, 126.0), (-19.0, 130.0), (-15.0, 234.0), (293.0, 203.0), (-306.0, 228.0), (-308.0, 114.0)],

    96: [(49.0, -243.0), (95.5, -242.5), (24.5, -195.5), (41.5, -216.0), (87.5, -220.5), (154.0, -205.0),
         (132.0, -243.0), (154.5, -228.0), (80.0, -196.5), (121.5, -197.0), (77.0, -163.5), (121.0, -163.0),
         (164.0, -120.0), (65.5, -125.0), (29.5, -143.0), (25.5, -116.0), (13.5, -84.0), (44.0, -64.5),
         (83.0, -106.5), (111.0, -105.5), (80.0, -65.5), (117.5, -65.5), (149.0, -93.0), (24.5, -163.5),
         (252.0, -242.0), (295.0, -241.5), (221.5, -195.5), (245.0, -216.0), (291.0, -219.5), (357.5, -203.5),
         (335.0, -240.5), (357.5, -227.0), (283.0, -196.5), (326.0, -195.5), (281.5, -163.0), (325.5, -162.5),
         (367.5, -120.5), (268.0, -125.0), (230.0, -142.5), (225.0, -115.5), (216.5, -81.5), (248.0, -64.0),
         (285.5, -105.0), (318.5, -105.0), (282.5, -64.0), (322.0, -64.0), (352.0, -93.0), (226.0, -164.0),
         (449.0, -243.0), (493.0, -243.0), (427.0, -196.5), (444.0, -216.0), (490.5, -219.5), (555.5, -205.0),
         (532.0, -242.5), (557.5, -224.5), (481.0, -197.0), (524.5, -196.0), (480.5, -163.5), (540.5, -162.5),
         (566.5, -121.0), (467.5, -126.0), (431.0, -143.5), (426.0, -114.5), (415.5, -83.5), (447.5, -64.0),
         (484.5, -106.0), (517.5, -106.0), (483.0, -64.0), (520.0, -64.5), (553.0, -94.0), (426.0, -164.0),
         (379.0, -28.0)],

    118:
        [(-403, -311), (-355, -311), (-380, -275), (-355, -245), (-369, -191), (-330, -193), (-299, -190), (-366, -88),
         (-364, -44), (-366, -7), (-320, -247), (-266, -266), (-241, -198), (-203, -231), (-188, -201), (-282, -153),
         (-221, -123), (-161, -123), (-131, -156), (-139, -142), (-131, -27), (-123, -3), (-131, 29), (-18, -46),
         (-162, 67), (-203, 39), (-324, 21), (-332, -15), (-331, -52), (-212, -88), (-292, -52), (-259, -29),
         (-4, -254), (32, -203), (-34, -148), (51, -155), (74, -221), (88, -127), (59, -265), (86, -296), (129, -296),
         (161, -296), (124, -198), (140, -226), (147, -163), (133, -138), (162, -134), (187, -173), (221, -125),
         (268, -215), (287, -225), (199, -258), (202, -296), (237, -295), (329, -296), (283, -297), (268, -248),
         (287, -248), (372, -277), (372, -197), (372, -153), (340, -74), (348, -254), (342, -168), (298, -29),
         (283, -74), (297, -92), (213, -62), (184, -50), (61, -45), (40, -73), (25, -52), (61, -84), (43, 53), (61, 73),
         (151, 73), (176, 99), (195, 53), (221, 33), (227, 73), (230, 56), (149, 131), (57, 154), (46, 171), (43, 205),
         (43, 229), (57, 245), (78, 205), (119, 207), (122, 241), (191, 243), (196, 207), (219, 186), (245, 154),
         (212, 154), (221, 132), (220, 110), (262, 94), (294, 74), (288, 154), (273, 230), (226, 229), (326, 230),
         (330, 152), (365, 154), (364, 91), (404, 154), (370, 191), (373, 212), (361, 253), (330, 260), (403, 253),
         (-256, -102), (-270, 0), (-236, 1), (229, -32), (-211, -266), (99, 74)]
}


class Renderer(object):
    def __init__(self, grid_case, or_ids, ex_ids, are_prods, are_loads):
        self.grid_case = grid_case
        self.grid_layout = np.asarray(case_layouts[grid_case])

        self.video_width, self.video_height = 1300, 800

        self.screen = pygame.display.set_mode((self.video_width, self.video_height), pygame.RESIZABLE)
        pygame.display.set_caption('Learning to Run a Power Network - render mode')  # Window title
        # Set default background color
        self.background_color = [70, 70, 73]
        self.screen.fill(self.background_color)

        self.topology_layout_shape = [1000, 800]
        self.topology_layout = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        # Substations layer
        self.nodes_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.nodes_outer_radius = 6
        self.nodes_inner_radius = 4
        #node_img = pygame.image.load(os.path.join(media_path, 'substation.png')).convert_alpha()
        #self.node_img = pygame.transform.scale(node_img, (20, 20))
        self.injections_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.are_prods = are_prods
        self.are_loads = are_loads

        # Lines layer
        self.lines_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.lines_ids_or = or_ids
        self.lines_ids_ex = ex_ids

        # Lines labels (e.g. mW) layer
        self.lines_labels_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA,
                                                   32).convert_alpha()

        self.left_menu_shape = [300, 800]
        self.left_menu = pygame.Surface(self.left_menu_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.left_menu_tile_color = [e + 10 for e in self.background_color]

        # Helpers for printing or plotting
        pygame.font.init()
        font = 'Arial'
        self.default_font = pygame.font.SysFont(font, 15)
        text_color = (180, 180, 180)
        value_color = (220, 220, 220)
        self.text_render = lambda s: self.default_font.render(s, False, text_color)
        self.value_render = lambda s: self.default_font.render(s, False, value_color)

        self.bold_white_font = pygame.font.SysFont(font, 15)
        bold_white = (220, 220, 220)
        self.bold_white_font.set_bold(True)
        self.bold_white_render = lambda s: self.bold_white_font.render(s, False, bold_white)
        # Containers for plotting prods and loads curves
        self.loads = []
        self.relative_thermal_limits = []

        self.game_over_font = pygame.font.SysFont("monospace", 40)
        red = (255, 0, 0)
        self.game_over_font.set_bold(True)
        self.game_over_render = lambda s: self.game_over_font.render(s, False, red)
        black = (0, 0, 0)
        self.game_over_shadow_render = lambda s: self.game_over_font.render(s, False, black)

    def draw_surface_nodes(self, scenario_id, date, prods, loads):
        self.loads.append(loads)
        surface = self.nodes_surface
        x_offset = int(self.topology_layout_shape[0] / 2.)
        y_offset = int(self.topology_layout_shape[1] / 2.)
        prods_iter = iter(prods)
        loads_iter = iter(loads)
        for i, ((x, y), is_prod, is_load) in enumerate(zip(self.grid_layout, self.are_prods, self.are_loads)):
            prod = next(prods_iter) if is_prod else 0.
            load = next(loads_iter) if is_load else 0.
            prod_minus_load = prod - load
            relative_prod = abs(prod / np.max(prods))
            relative_load = abs(load / np.max(loads))
            if prod_minus_load > 0:
                color = (0, 153, 255)
                offset_radius = 0 if relative_prod < 0.4 else 2 if relative_prod < 0.7 else 4
            elif prod_minus_load < 0:
                color = (210, 77, 255)
                offset_radius = 0 if relative_load < 0.4 else 2 if relative_load < 0.7 else 4
            else:
                color = (255, 255, 255)
                offset_radius = 0

            gfxdraw.aacircle(surface, x + x_offset, y + y_offset, self.nodes_outer_radius + offset_radius, color)
            gfxdraw.filled_circle(surface, x + x_offset, y + y_offset, self.nodes_outer_radius + offset_radius, color)
            gfxdraw.aacircle(surface, x + x_offset, y + y_offset, self.nodes_inner_radius, self.background_color)
            gfxdraw.filled_circle(surface, x + x_offset, y + y_offset, self.nodes_inner_radius, self.background_color)

        # Print some scenario stats
        surface.blit(self.text_render('Scenario id'), (80, 10))
        surface.blit(self.value_render(str(scenario_id)), (200, 10))
        surface.blit(self.text_render('Date'), (330, 10))
        surface.blit(self.value_render(date.strftime("%a, %d %b %H:%M")), (400, 10))

    def draw_surface_lines(self, relative_thermal_limits, lines_por, lines_service_status):
        def draw_arrow_head(x, y, angle, color, thickness):
            head_angle = math.pi / 8.
            width = 8 + thickness
            x -= width / 2. * math.cos(angle)
            y -= width / 2. * math.sin(angle)
            x1 = x + width * math.cos(angle + head_angle)
            y1 = y + width * math.sin(angle + head_angle)
            x2 = x + width * math.cos(angle - head_angle)
            y2 = y + width * math.sin(angle - head_angle)
            gfxdraw.aapolygon(surface, ((x, y), (x1, y1), (x2, y2)), color)
            gfxdraw.filled_polygon(surface, ((x, y), (x1, y1), (x2, y2)), color)

        def draw_dashed_line(ori, ext):
            length_x = ori[0] - ext[0]
            length_y = ori[1] - ext[1]
            total_length = math.sqrt((ori[0] - ext[0]) ** 2. + (ori[1] - ext[1]) ** 2.) - 2. * self.nodes_outer_radius
            center = ((ori[0] + ext[0]) / 2., (ori[1] + ext[1]) / 2.)
            angle = math.atan2(ori[1] - ext[1], ori[0] - ext[0])

            tick_length = 5
            grey_color = (200, 200, 200)
            thickness = 1
            n_ticks = int(total_length / (2 * tick_length)) + 1
            for i in range(n_ticks):
                tick_center = (center[0] + (i - n_ticks / 2.) / n_ticks * length_x,
                               center[1] + (i - n_ticks / 2.) / n_ticks * length_y)
                UL = (x_offset + tick_center[0] + (tick_length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
                      y_offset + tick_center[1] + (thickness / 2.) * math.cos(angle) + (tick_length / 2.) * math.sin(angle))
                UR = (x_offset + tick_center[0] - (tick_length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
                      y_offset + tick_center[1] + (thickness / 2.) * math.cos(angle) - (tick_length / 2.) * math.sin(angle))
                BL = (x_offset + tick_center[0] + (tick_length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
                      y_offset + tick_center[1] - (thickness / 2.) * math.cos(angle) + (tick_length / 2.) * math.sin(angle))
                BR = (x_offset + tick_center[0] - (tick_length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
                      y_offset + tick_center[1] - (thickness / 2.) * math.cos(angle) - (tick_length / 2.) * math.sin(angle))
                gfxdraw.aapolygon(surface, (UL, UR, BR, BL), grey_color)
                gfxdraw.filled_polygon(surface, (UL, UR, BR, BL), grey_color)

        self.relative_thermal_limits.append(relative_thermal_limits)
        surface = self.lines_surface
        layout = self.grid_layout
        x_offset = int(self.topology_layout_shape[0] / 2.)
        y_offset = int(self.topology_layout_shape[1] / 2.)
        for or_id, ex_id, rtl, line_por, is_on in zip(self.lines_ids_or, self.lines_ids_ex, relative_thermal_limits,
                                                      lines_por, lines_service_status):
            # If line disconnected, call special drawing function
            if not is_on:
                draw_dashed_line(layout[or_id], layout[ex_id])
                continue

            # Otherwise, plot one straight line, with custom width, color, and arrow heads
            thickness = 1 if rtl < .3 else 2 if rtl < .7 else 4
            color = (51, 204, 51) if rtl < .9 else (255, 165, 0) if rtl < 1. else (214, 0, 0)
            if line_por >= 0:
                ori = layout[or_id]
                ext = layout[ex_id]
            else:
                ori = layout[ex_id]
                ext = layout[or_id]
            length = math.sqrt((ori[0] - ext[0]) ** 2. + (ori[1] - ext[1]) ** 2.) - 2. * self.nodes_outer_radius
            center = ((ori[0] + ext[0]) / 2., (ori[1] + ext[1]) / 2.)
            angle = math.atan2(ori[1] - ext[1], ori[0] - ext[0])
            UL = (x_offset + center[0] + (length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
                  y_offset + center[1] + (thickness / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
            UR = (x_offset + center[0] - (length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
                  y_offset + center[1] + (thickness / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))
            BL = (x_offset + center[0] + (length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
                  y_offset + center[1] - (thickness / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
            BR = (x_offset + center[0] - (length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
                  y_offset + center[1] - (thickness / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))
            #pygame.draw.aaline(surface, color, start, end, 0)
            pygame.draw.aalines(surface, color, 1, (UL, UR, BR, BL), 0)
            gfxdraw.aapolygon(surface, (UL, UR, BR, BL), color)
            gfxdraw.filled_polygon(surface, (UL, UR, BR, BL), color)
            #pygame.draw.polygon(surface, color, (UL, UR, BR, BL), 0)
            #gfxdraw.aapolygon(surface, (UL, UR, BR, BL), color)
            #gfxdraw.filled_polygon(surface, (UL, UR, BR, BL), color)
            #gfxdraw.line(surface, x_or + x_offset, y_or + y_offset, x_ex + x_offset, y_ex + y_offset, (255, 255, 255))

            distance_arrow_heads = 30
            n_arrow_heads = int(max(1, length // distance_arrow_heads))
            for a in range(n_arrow_heads):
                if n_arrow_heads != 1:
                    x = x_offset + center[0] + ((a + .5) * distance_arrow_heads - length / 2.) * math.cos(angle)
                    y = y_offset + center[1] + ((a + .5) * distance_arrow_heads - length / 2.) * math.sin(angle)
                else:
                    x = x_offset + center[0]
                    y = y_offset + center[1]
                draw_arrow_head(x, y, angle, color, thickness)

    def create_plot_loads_curve(self, n_hours, left_xlabel):
        facecolor_asfloat = np.asarray(self.left_menu_tile_color) / 255.
        layout_config = {'pad': 0.2}
        fig = pylab.figure(figsize=[3, 1.5], dpi=100, facecolor=facecolor_asfloat, tight_layout=layout_config)
        ax = fig.gca()
        # Retrieve data for the specified time
        data = np.sum(self.loads, axis=-1)
        data = data[-min(len(data), n_hours):]
        n_data = len(data)
        ax.plot(np.linspace(n_data, 0, num=n_data), data, '#d24dff')
        # Ticks and labels
        ax.set_xlim([n_hours, 1])
        ax.set_xticks([1, n_hours])
        ax.set_xticklabels(['now', left_xlabel])
        ax.set_ylim([0, np.max(data) * 1.05])
        ax.set_yticks([0, np.max(data)])
        ax.set_yticklabels(['', '%.0fGW' % (np.max(data) / 1000.)])
        label_color_hexa = '#a6a6a6'
        ax.tick_params(axis='y', labelsize=6, pad=-27, labelcolor=label_color_hexa)
        ax.tick_params(axis='x', labelsize=6, bottom=False, labelcolor=label_color_hexa)
        # Top and right axis
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_facecolor(np.asarray(self.background_color) / 255.)
        fig.tight_layout()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        return pygame.image.fromstring(raw_data, size, "RGB")

    def create_plot_relative_thermal_limits(self, n_hours, left_xlabel):
        facecolor_asfloat = np.asarray(self.left_menu_tile_color) / 255.
        layout_config = {'pad': 0.2}
        fig = pylab.figure(figsize=[3, 1.5], dpi=100, facecolor=facecolor_asfloat, tight_layout=layout_config)
        ax = fig.gca()
        # Retrieve data for the specified time
        data = self.relative_thermal_limits
        data = data[-min(len(data), n_hours):]
        n_data = len(data)
        medians = np.median(data, axis=-1)
        percentiles_90 = np.percentile(data, 90, axis=-1)
        percentiles_10 = np.percentile(data, 10, axis=-1)
        ax.plot(np.linspace(n_data, 0, num=n_data), medians, '#84e184')
        ax.fill_between(np.linspace(n_data, 0, num=n_data), percentiles_10, percentiles_90, color='#239023')
        # ax.plot(np.linspace(n_data, 0, num=n_data), percentiles_10, '#33cc33')
        # ax.plot(np.linspace(n_data, 0, num=n_data), percentiles_90, '#33cc33')
        # Ticks and labels
        ax.set_xlim([n_hours, 1])
        ax.set_xticks([1, n_hours])
        ax.set_xticklabels(['now', left_xlabel])
        ax.set_ylim([0, max(1.05, np.max([medians, percentiles_90, percentiles_10]) * 1.05)])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['', '1'])
        label_color_hexa = '#a6a6a6'
        ax.tick_params(axis='y', labelsize=6, pad=-12, labelcolor=label_color_hexa)
        ax.tick_params(axis='x', labelsize=6, bottom=False, labelcolor=label_color_hexa)
        # Top and right axis
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_facecolor(np.asarray(self.background_color) / 255.)
        fig.tight_layout()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        return pygame.image.fromstring(raw_data, size, "RGB")

    def create_plot_number_overflows(self, n_hours, left_xlabel):
        facecolor_asfloat = np.asarray(self.left_menu_tile_color) / 255.
        layout_config = {'pad': 0.2}
        fig = pylab.figure(figsize=[3, 1], dpi=100, facecolor=facecolor_asfloat, tight_layout=layout_config)
        ax = fig.gca()
        # Retrieve data for the specified time
        data = np.sum(np.asarray(self.relative_thermal_limits) >= 1., axis=-1)
        data = data[-min(len(data), n_hours):]
        n_data = len(data)
        ax.plot(np.linspace(n_data, 0, num=n_data), data, '#ff3333')
        # Ticks and labels
        ax.set_xlim([n_hours, 1])
        ax.set_xticks([1, n_hours])
        ax.set_xticklabels(['now', left_xlabel])
        ax.set_ylim([0, max(1, np.max(data) * 1.05)])
        ax.set_yticks([0, max(1, np.max(data))])
        ax.set_yticklabels(['', '%d' % np.max(data)])
        label_color_hexa = '#a6a6a6'
        ax.tick_params(axis='y', labelsize=6, pad=-12, labelcolor=label_color_hexa)
        ax.tick_params(axis='x', labelsize=6, bottom=False, labelcolor=label_color_hexa)
        # Top and right axis
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_facecolor(np.asarray(self.background_color) / 255.)
        fig.tight_layout()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        return pygame.image.fromstring(raw_data, size, "RGB")

    def draw_surface_rewards(self, rewards):
        last_rewards_surface_shape = (self.left_menu_shape[0], 160)
        last_rewards_surface = pygame.Surface(last_rewards_surface_shape, pygame.SRCALPHA, 32).convert_alpha()
        last_rewards_surface.fill(self.left_menu_tile_color)
        last_rewards_surface.blit(self.bold_white_render('Last timestep reward'), (30, 20))

        reward_offset = (50, 40)
        x_offset = 180
        line_spacing = 20
        rewards_labels = ['Line capacity usage', 'Cost of action', 'Distance to initial grid', 'Connexity valuation']
        for i, (reward, label) in enumerate(zip(rewards, rewards_labels)):
            last_rewards_surface.blit(self.text_render(label), (reward_offset[0], reward_offset[1] + i * line_spacing))
            last_rewards_surface.blit(self.value_render('%.1f' % reward),
                                      (reward_offset[0] + x_offset, reward_offset[1] + i * line_spacing))
        last_rewards_surface.blit(self.text_render('Total'),
                                  (reward_offset[0], reward_offset[1] + (i + 1) * line_spacing))
        last_rewards_surface.blit(self.value_render('%.1f' % np.sum(rewards)),
                                  (reward_offset[0] + x_offset, reward_offset[1] + (i + 1) * line_spacing))

        gfxdraw.hline(last_rewards_surface, 0, last_rewards_surface_shape[0], 0, (64, 64, 64))
        gfxdraw.hline(last_rewards_surface, 0, last_rewards_surface_shape[0], last_rewards_surface_shape[1] - 1,
                      (64, 64, 64))
        return last_rewards_surface

    def draw_surface_loads_curves(self):
        # Loads curve surface: retrieve images surfaces, stack them into a common surface, plot horizontal lines
        # at top and bottom of latter surface
        img_loads_curve_week = self.create_plot_loads_curve(n_hours=7 * 24, left_xlabel=' 7 days ago  ')
        img_loads_curve_day = self.create_plot_loads_curve(n_hours=24, left_xlabel='24 hours ago')
        loads_curve_surface = pygame.Surface(
            (img_loads_curve_week.get_width(), 2 * img_loads_curve_week.get_height() + 30),
            pygame.SRCALPHA, 32).convert_alpha()
        loads_curve_surface.fill(self.left_menu_tile_color)
        loads_curve_surface.blit(self.bold_white_render('Total demand'), (30, 10))
        loads_curve_surface.blit(img_loads_curve_week, (0, 30))
        loads_curve_surface.blit(img_loads_curve_day, (0, 30 + img_loads_curve_week.get_height()))
        gfxdraw.hline(loads_curve_surface, 0, loads_curve_surface.get_width(), 0, (64, 64, 64))
        gfxdraw.hline(loads_curve_surface, 0, loads_curve_surface.get_width(), loads_curve_surface.get_height() - 1,
                      (64, 64, 64))

        return loads_curve_surface

    def draw_surface_relative_thermal_limits(self):
        img_rtl = self.create_plot_relative_thermal_limits(n_hours=24, left_xlabel='24 hours ago')
        rtl_curves_surface = pygame.Surface((img_rtl.get_width(), 2 * img_rtl.get_height() + 30),
                                            pygame.SRCALPHA, 32).convert_alpha()
        rtl_curves_surface.fill(self.left_menu_tile_color)
        rtl_curves_surface.blit(self.bold_white_render('Lines capacity usage'), (30, 10))
        rtl_curves_surface.blit(img_rtl, (0, 30))
        gfxdraw.hline(rtl_curves_surface, 0, rtl_curves_surface.get_width(), 0, (64, 64, 64))
        gfxdraw.hline(rtl_curves_surface, 0, rtl_curves_surface.get_width(), rtl_curves_surface.get_height() - 1,
                      (64, 64, 64))

        return rtl_curves_surface

    def draw_surface_n_overflows(self):
        img_rtl = self.create_plot_number_overflows(n_hours=7 * 24, left_xlabel=' 7 days ago  ')
        n_overflows_surface = pygame.Surface((img_rtl.get_width(), 2 * img_rtl.get_height() + 30),
                                             pygame.SRCALPHA, 32).convert_alpha()
        n_overflows_surface.fill(self.left_menu_tile_color)
        n_overflows_surface.blit(self.bold_white_render('Number of overflows'), (30, 10))
        n_overflows_surface.blit(img_rtl, (0, 30))
        gfxdraw.hline(n_overflows_surface, 0, n_overflows_surface.get_width(), 0, (64, 64, 64))
        gfxdraw.hline(n_overflows_surface, 0, n_overflows_surface.get_width(), n_overflows_surface.get_height() - 1,
                      (64, 64, 64))

        return n_overflows_surface

    def draw_surface_legend(self):
        surface_shape = (self.left_menu_shape[0], 100)
        surface = pygame.Surface(surface_shape, pygame.SRCALPHA, 32).convert_alpha()
        surface.fill(self.left_menu_tile_color)
        surface.blit(self.bold_white_render('Legend'), (15, 10))

        # Lines legend
        xs, ys, thi, lrg = 30, 30, 1, 40
        gfxdraw.filled_polygon(surface, ((xs, ys), (xs + lrg, ys), (xs + lrg, ys + thi), (xs, ys + thi)), (51, 204, 51))
        xs, ys, thi, lrg = xs + lrg, 30, 2, 40
        gfxdraw.filled_polygon(surface, ((xs, ys), (xs + lrg, ys), (xs + lrg, ys + thi), (xs, ys + thi)), (51, 204, 51))
        xs, ys, thi, lrg = xs + lrg, 30, 4, 40
        gfxdraw.filled_polygon(surface, ((xs, ys), (xs + lrg, ys), (xs + lrg, ys + thi), (xs, ys + thi)), (51, 204, 51))
        
        return surface

    def draw_plot_game_over(self):
        game_over_surface = pygame.Surface((500, 200), pygame.SRCALPHA, 32).convert_alpha()
        game_over_text = 'Game over!'
        game_over_surface.blit(self.game_over_shadow_render(game_over_text), (2, 2))
        game_over_surface.blit(self.game_over_render(game_over_text), (0, 0))

        return game_over_surface

    def _update_left_menu(self, epoch, timestep, rewards):
        self.left_menu = pygame.Surface(self.left_menu_shape, pygame.SRCALPHA, 32).convert_alpha()

        # Top info about epoch and timestep
        self.left_menu.blit(self.text_render('Epoch'), (30, 10))
        self.left_menu.blit(self.text_render('Timestep'), (150, 10))
        self.left_menu.blit(self.value_render(str(epoch)), (100, 10))
        self.left_menu.blit(self.value_render(str(timestep)), (250, 10))

        # Last reward surface
        #last_rewards_surface = self.draw_surface_rewards(rewards)

        # Loads curve surface
        loads_curve_surface = self.draw_surface_loads_curves()

        # Relative thermal limits curves
        rtl_curves_surface = self.draw_surface_relative_thermal_limits()

        # Number of overflowed lines curves
        n_overflows_surface = self.draw_surface_n_overflows()

        gfxdraw.vline(self.left_menu, self.left_menu_shape[0] - 1, 0, self.left_menu_shape[1], (128, 128, 128))
        #self.left_menu.blit(last_rewards_surface, (0, 50))
        self.left_menu.blit(loads_curve_surface, (0, 50))
        self.left_menu.blit(rtl_curves_surface, (0, 380))
        self.left_menu.blit(n_overflows_surface, (0, 560))

    # noinspection PyArgumentList
    def _update_topology(self, scenario_id, date, relative_thermal_limits, lines_por, lines_service_status,
                         prods, loads, rewards, game_over):
        self.topology_layout = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.nodes_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.injections_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.lines_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        gfxdraw.vline(self.topology_layout, 0, 0, self.left_menu_shape[1], (20, 20, 20))

        # Lines
        self.draw_surface_lines(relative_thermal_limits, lines_por, lines_service_status)

        # Injections
        last_rewards_surface = self.draw_surface_rewards(rewards)

        # Legend
        #legend_surface = self.draw_surface_legend()

        # Nodes
        self.draw_surface_nodes(scenario_id, date, prods, loads)

        self.topology_layout.blit(self.lines_surface, (0, 0))
        self.topology_layout.blit(last_rewards_surface, (1, 570))
        #self.topology_layout.blit(legend_surface, (1, 470))
        self.topology_layout.blit(self.nodes_surface, (0, 0))

        # Print a game over message if game has been lost
        if game_over:
            game_over_surface = self.draw_plot_game_over()
            self.topology_layout.blit(game_over_surface, (300, 200))

    def render(self, relative_thermal_limits, lines_por, lines_service_status,
               epoch, timestep, scenario_id, prods, loads, last_timestep_rewards, date, game_over=False):
        plt.close('all')
        self.screen.fill(self.background_color)

        # Execute full ploting mechanism: order is important
        self._update_topology(scenario_id, date, relative_thermal_limits, lines_por, lines_service_status,
                              prods, loads, last_timestep_rewards, game_over=game_over)
        self._update_left_menu(epoch, timestep, last_timestep_rewards)

        # Blit all macro surfaces on screen
        self.screen.blit(self.topology_layout, (self.left_menu_shape[0], 0))
        self.screen.blit(self.left_menu, (0, 0))

        pygame.display.flip()
        # Bugfix for mac
        pygame.event.get()


def scale(u, t):
    for k, v in case_layouts.items():
        print(k)
        print([(int(a * u + +20), int(b * u + +50)) for a, b in v])


def recenter():
    for k, v in case_layouts.items():
        print(k)
        arr = np.asarray(np.absolute(v))
        minix = np.min(arr[:, 0])
        miniy = np.min(arr[:, 1])
        maxix = np.max(arr[:, 0])
        maxiy = np.max(arr[:, 1])

        x = (maxix - minix) / 2.
        y = (maxiy - miniy) / 2.
        print([(int(a - x), int(-b - y)) for a, b in v])


if __name__ == '__main__':
    scale(1., -10)