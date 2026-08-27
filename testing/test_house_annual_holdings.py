'''House annual disclosures: a PDF table read from its own geometry.

Every fixture here is verbatim from a real filing. The geometry is Greg
Landsman's CY2025 annual report (document 10074384) and Patti Adair's
Candidate report (10073223), captured rectangle by rectangle and word by
word. The NUL bytes in the location and description labels are the real
bytes the PDF produces, not spaces transcribed to look like them: `\\s` does
not match NUL, so a fixture that quietly turned them into spaces would go
green against text no filing emits.

The five things this file exists to pin down:

* `None` in the value column is an asset **sold during the year** that still
  paid out over $200. It is an exited position, not a holding worth nothing.
* `Undetermined` has no bounds at all. Zero would be a different disclosure.
* `Over $50,000,000` and `Spouse/DC Over $1,000,000` are floors with no
  ceiling. An upper bound equal to the lower one makes a portfolio whose
  minimum exceeds its maximum.
* An entry cut in half by a page break is one holding, not two. Left
  unstitched it arrives as a row with a value and no asset, followed by a row
  with an asset and no value -- both of which look perfectly plausible.
* A location or description strip runs the full width of the table. Left in,
  its words scatter into whichever column they happen to sit under.
'''
import os

import pytest

from testing._gates import requires_sec
from tools.altdata_server import house_annual as ha
from tools.altdata_server.congress_trades import DisclosureUnavailable

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"

# The House Clerk asks callers to identify themselves, so these need the same
# SEC_EMAIL contact the rest of the disclosure code does. `requires_sec` also
# loads .env, which pytest does not do on its own.
live = pytest.mark.skipif(SKIP_NETWORK, reason="SKIP_NETWORK_TESTS=1")


def _page(rects, words, height=792.0, width=612.0, text=""):
    """The shape `ha.page_geometry()` hands the table reader."""
    return {
        "width": width, "height": height, "text": text,
        "rects": [{"x0": x0, "x1": x1, "top": top, "bottom": bottom,
                   "grey": grey} for x0, x1, top, bottom, grey in rects],
        "words": [{"text": t, "x0": x0, "x1": x1, "top": top}
                  for t, x0, x1, top in words],
    }


# The filer block of 10074384, verbatim -- NUL bytes and all.
COVER_TEXT = (
    'Filing ID #10074384\n'
    'F\x00\x00\x00\x00\x00\x00\x00\x00 D\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    ' R\x00\x00\x00\x00\x00\n'
    'Clerk of the House of Representatives • Legislative Resource Center'
    ' • B81 Cannon Building • Washington, DC 20515\n'
    'F\x00\x00\x00\x00 I\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n'
    'Name: Hon. Greg Landsman\n'
    'Status: Member\n'
    'State/District: OH01\n'
    'F\x00\x00\x00\x00\x00 I\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n'
    'Filing Type: Annual Report\n'
    'Filing Year: 2025\n'
    'Filing Date: 04/20/2026\n'
    'S\x00\x00\x00\x00\x00\x00\x00 A: A\x00\x00\x00\x00\x00 \x00\x00\x00 '
    '"U\x00\x00\x00\x00\x00\x00\x00" I\x00\x00\x00\x00\x00\n'
)

# The trailing section of 10074384 that defines every account appearing on
# the left of a `⇒`, verbatim.
VEHICLE_TEXT = (
    'S\x00\x00\x00\x00\x00\x00\x00 A \x00\x00\x00 B I\x00\x00\x00\x00\x00\x00'
    '\x00\x00\x00 V\x00\x00\x00\x00\x00\x00 D\x00\x00\x00\x00\x00\x00\n'
    'Rockefeller Capital Management (1) (100% Interest)\n'
    'Greg Landman Roth IRA\n'
    'Sarah Landsman Traditional IRA (Owner: SP)\n'
    'Rockefeller Capital Management (2) (Owner: JT)\n'
    'Fidelity - 529 Plan (Owner: DC)\n'
    'L\x00\x00\x00\x00\x00\x00\x00: MA\n'
    'Fidelity Brokerage (Owner: SP)\n'
    'Rockefeller Capital Management (1) (Owner: JT)\n'
    'Kroger - 401(k) (Owner: SP)\n'
    'BLK CollegeAdvantage 529 Plan - DC #1\n'
    'L\x00\x00\x00\x00\x00\x00\x00: OH\n'
    'BLK CollegeAdvantage 529 Plan - DC #2\n'
    'L\x00\x00\x00\x00\x00\x00\x00: OH\n'
    'E\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x00 S\x00\x00\x00\x00\x00, '
    'D\x00\x00\x00\x00\x00\x00\x00\x00, \x00\x00 T\x00\x00\x00\x00 '
    'I\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n'
    'IPO: Did you purchase any shares that were allocated as a part of an '
    'Initial Public Offering?\n'
)

# The filer block of 10073223, verbatim.
ADAIR_COVER_TEXT = (
    'Filing ID #10073223\n'
    'Name: Patti Adair\n'
    'Status: Congressional Candidate\n'
    'State/District: OR05\n'
    'Filing Type: Candidate Report\n'
    'Filing Year: 2025\n'
    'Filing Date: 03/17/2026\n'
    'Period Covered: 01/01/2024– 11/18/2025\n'
)

# 10074384 page 1: the last two entries. The second runs off the bottom
# of the page carrying its location strip with it.
_LANDSMAN_PAGE_1 = _page(
    rects=[
        (21.75, 264.0, 360.0, 392.25, 0.9333),
        (264.0, 303.0, 360.0, 392.25, 0.9333),
        (303.0, 385.5, 360.0, 392.25, 0.9333),
        (385.5, 468.0, 360.0, 392.25, 0.9333),
        (468.0, 531.0, 360.0, 392.25, 0.9333),
        (531.0, 576.0, 360.0, 392.25, 0.9333),
        (21.75, 264.0, 624.75, 658.5, 0.9608),
        (264.0, 303.0, 624.75, 658.5, 0.9608),
        (303.0, 385.5, 624.75, 658.5, 0.9608),
        (385.5, 468.0, 624.75, 658.5, 0.9608),
        (468.0, 531.0, 624.75, 658.5, 0.9608),
        (531.0, 576.0, 624.75, 658.5, 0.9608),
        (21.75, 576.0, 658.5, 692.25, 0.9608),
    ],
    words=[
        ('Asset', 25.32, 51.34, 366.87),
        ('Owner', 266.82, 299.57, 366.87),
        ('Value', 305.82, 333.71, 366.87),
        ('of', 336.11, 345.83, 366.87),
        ('Asset', 348.23, 374.24, 366.87),
        ('Income', 388.32, 425.09, 366.87),
        ('Type(s)', 427.49, 464.17, 366.87),
        ('Income', 470.82, 507.59, 366.87),
        ('Tx.', 533.82, 548.95, 366.87),
        ('>', 551.34, 557.99, 366.87),
        ('$1,000?', 533.82, 572.67, 378.12),
        ('BLK', 25.2, 42.76, 631.97),
        ('CollegeAdvantage', 44.93, 116.1, 631.97),
        ('529', 118.27, 133.15, 631.97),
        ('Plan', 135.32, 153.23, 631.97),
        ('-', 155.41, 158.77, 631.97),
        ('DC', 160.94, 173.46, 631.97),
        ('#2', 175.63, 186.45, 631.97),
        ('$100,001', 305.7, 343.44, 631.97),
        ('-', 345.61, 348.98, 631.97),
        ('Tax-Deferred', 388.2, 441.47, 631.97),
        ('⇒', 188.62, 196.41, 632.0),
        ('$250,000', 305.7, 345.49, 642.47),
        ('BR', 25.2, 37.39, 643.22),
        ('COLLEGE', 39.57, 81.19, 643.22),
        ('2032', 83.36, 103.91, 643.22),
        ('OPTION', 106.08, 140.94, 643.22),
        ('-', 143.11, 146.48, 643.22),
        ('A', 148.65, 154.68, 643.22),
        ('[5F]', 156.86, 173.76, 643.22),
        ('L\x00\x00\x00\x00\x00\x00\x00:', 25.05, 64.19, 660.82),
        ('OH', 66.26, 79.59, 660.82),
        ('D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00:', 25.05, 76.18, 673.57),
        ('Asset', 78.25, 98.46, 673.57),
        ('transferred', 100.52, 142.99, 673.57),
        ('from', 145.05, 163.47, 673.57),
        ('filer’s', 165.54, 186.54, 673.57),
        ('relative', 188.6, 216.82, 673.57),
        ('in', 218.89, 226.44, 673.57),
        ('2025.', 228.51, 250.13, 673.57),
        ('Fidelity', 25.2, 55.49, 699.47),
        ('-', 57.66, 61.03, 699.47),
        ('529', 63.2, 78.07, 699.47),
        ('Plan', 80.25, 98.16, 699.47),
        ('DC', 266.7, 279.22, 699.47),
        ('$15,001', 305.7, 337.15, 699.47),
        ('-', 339.32, 342.68, 699.47),
        ('$50,000', 344.85, 379.62, 699.47),
        ('Tax-Deferred', 388.2, 441.47, 699.47),
        ('⇒', 100.33, 108.12, 699.5),
        ('MA', 25.2, 39.58, 710.72),
        ('College', 41.75, 70.81, 710.72),
        ('Portfolio', 72.99, 107.96, 710.72),
        ('[5F]', 110.14, 127.03, 710.72),
    ],
    text=COVER_TEXT,
)

# 10074384 page 2: the tail of the entry above, then three nested
# holdings inside a Fidelity brokerage account.
_LANDSMAN_PAGE_2 = _page(
    rects=[
        (21.75, 264.0, 72.0, 104.25, 0.9333),
        (264.0, 303.0, 72.0, 104.25, 0.9333),
        (303.0, 385.5, 72.0, 104.25, 0.9333),
        (385.5, 468.0, 72.0, 104.25, 0.9333),
        (468.0, 531.0, 72.0, 104.25, 0.9333),
        (531.0, 576.0, 72.0, 104.25, 0.9333),
        (21.75, 264.0, 131.25, 165.0, 0.9608),
        (264.0, 303.0, 131.25, 165.0, 0.9608),
        (303.0, 385.5, 131.25, 165.0, 0.9608),
        (385.5, 468.0, 131.25, 165.0, 0.9608),
        (468.0, 531.0, 131.25, 165.0, 0.9608),
        (531.0, 576.0, 131.25, 165.0, 0.9608),
        (21.75, 576.0, 165.0, 173.25, 0.9608),
        (21.75, 264.0, 228.0, 261.75, 0.9608),
        (264.0, 303.0, 228.0, 261.75, 0.9608),
        (303.0, 385.5, 228.0, 261.75, 0.9608),
        (385.5, 468.0, 228.0, 261.75, 0.9608),
        (468.0, 531.0, 228.0, 261.75, 0.9608),
        (531.0, 576.0, 228.0, 261.75, 0.9608),
        (21.75, 576.0, 261.75, 295.5, 0.9608),
    ],
    words=[
        ('Asset', 25.32, 51.34, 78.87),
        ('Owner', 266.82, 299.57, 78.87),
        ('Value', 305.82, 333.71, 78.87),
        ('of', 336.11, 345.83, 78.87),
        ('Asset', 348.23, 374.24, 78.87),
        ('Income', 388.32, 425.09, 78.87),
        ('Type(s)', 427.49, 464.17, 78.87),
        ('Income', 470.82, 507.59, 78.87),
        ('Tx.', 533.82, 548.95, 78.87),
        ('>', 551.34, 557.99, 78.87),
        ('$1,000?', 533.82, 572.67, 90.12),
        ('L\x00\x00\x00\x00\x00\x00\x00:', 25.05, 64.19, 112.57),
        ('MA', 66.26, 79.92, 112.57),
        ('Fidelity', 25.2, 55.49, 138.47),
        ('Brokerage', 57.66, 98.41, 138.47),
        ('SP', 266.7, 277.23, 138.47),
        ('$1', 305.7, 315.05, 138.47),
        ('-', 317.22, 320.59, 138.47),
        ('$1,000', 322.76, 351.11, 138.47),
        ('Dividends', 388.2, 428.57, 138.47),
        ('$201', 470.7, 490.6, 138.47),
        ('-', 492.77, 496.14, 138.47),
        ('$1,000', 498.31, 526.66, 138.47),
        ('⇒', 100.58, 108.37, 138.5),
        ('Fidelity', 25.2, 55.49, 149.72),
        ('Government', 57.66, 107.57, 149.72),
        ('Money', 109.74, 137.03, 149.72),
        ('Market', 139.2, 168.05, 149.72),
        ('Fund', 170.22, 191.27, 149.72),
        ('(SPAXX)', 193.44, 229.56, 149.72),
        ('[MF]', 231.73, 252.22, 149.72),
        ('Fidelity', 25.2, 55.49, 180.47),
        ('Brokerage', 57.66, 98.41, 180.47),
        ('SP', 266.7, 277.23, 180.47),
        ('None', 305.7, 327.12, 180.47),
        ('Dividends', 388.2, 428.57, 180.47),
        ('$1,001', 470.7, 497.39, 180.47),
        ('-', 499.56, 502.93, 180.47),
        ('⇒', 100.58, 108.37, 180.5),
        ('$2,500', 470.7, 499.44, 190.97),
        ('Kroger', 25.2, 52.6, 191.72),
        ('Company', 54.77, 92.76, 191.72),
        ('(KR)', 94.93, 114.24, 191.72),
        ('[ST]', 116.42, 133.78, 191.72),
        ('D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00:', 25.05, 76.18, 209.32),
        ('Stock', 78.25, 99.07, 209.32),
        ('transferred', 101.13, 143.6, 209.32),
        ('to', 145.66, 153.22, 209.32),
        ('Rockefeller', 155.28, 197.93, 209.32),
        ('Capital', 199.99, 226.89, 209.32),
        ('Management', 228.95, 278.7, 209.32),
        ('(1)', 280.76, 290.85, 209.32),
        ('Fidelity', 25.2, 55.49, 235.22),
        ('Brokerage', 57.66, 98.41, 235.22),
        ('SP', 266.7, 277.23, 235.22),
        ('None', 305.7, 327.12, 235.22),
        ('None', 388.2, 409.62, 235.22),
        ('⇒', 100.58, 108.37, 235.25),
        ('Kroger', 25.2, 52.6, 246.47),
        ('Company', 54.77, 92.76, 246.47),
        ('(KR)', 94.93, 114.24, 246.47),
        ('-', 116.41, 119.78, 246.47),
        ('Restricted', 121.95, 162.65, 246.47),
        ('Stock', 164.82, 186.73, 246.47),
        ('[OT]', 188.91, 207.92, 246.47),
        ('D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00:', 25.05, 76.18, 264.07),
        ('Employee', 78.25, 115.78, 264.07),
        ('Benefit', 117.84, 144.98, 264.07),
        ('C\x00\x00\x00\x00\x00\x00\x00:', 25.05, 69.3, 276.82),
        ('Shares', 71.37, 96.78, 276.82),
        ('were', 98.84, 116.91, 276.82),
        ('forfeited', 118.98, 151.28, 276.82),
        ('when', 153.34, 173.81, 276.82),
        ('spouse', 175.87, 201.8, 276.82),
        ('terminated', 203.87, 245.84, 276.82),
        ('her', 247.91, 260.52, 276.82),
        ('employment', 262.58, 310.06, 276.82),
        ('with', 312.12, 328.86, 276.82),
        ('84.51', 330.92, 351.35, 276.82),
        ('(owned', 353.41, 381.62, 276.82),
        ('by', 383.68, 392.68, 276.82),
        ('Kroger)', 394.74, 423.99, 276.82),
        ('in', 426.05, 433.61, 276.82),
        ('2025.', 435.67, 457.29, 276.82),
    ],
    text=VEHICLE_TEXT,
)

# 10073223 page 1: a Candidate report, which drops the `Tx. > $1,000?`
# column and carries two income columns in its place.
_ADAIR_PAGE_1 = _page(
    rects=[
        (21.75, 238.5, 375.75, 414.75, 0.9333),
        (238.5, 277.5, 375.75, 414.75, 0.9333),
        (277.5, 360.0, 375.75, 414.75, 0.9333),
        (360.0, 442.5, 375.75, 414.75, 0.9333),
        (442.5, 513.0, 375.75, 414.75, 0.9333),
        (513.0, 576.0, 375.75, 414.75, 0.9333),
        (21.75, 238.5, 482.25, 504.75, 0.9608),
        (238.5, 277.5, 482.25, 504.75, 0.9608),
        (277.5, 360.0, 482.25, 504.75, 0.9608),
        (360.0, 442.5, 482.25, 504.75, 0.9608),
        (442.5, 513.0, 482.25, 504.75, 0.9608),
        (513.0, 576.0, 482.25, 504.75, 0.9608),
        (21.75, 576.0, 504.75, 513.0, 0.9608),
        (21.75, 238.5, 580.5, 624.75, 0.9608),
        (238.5, 277.5, 580.5, 624.75, 0.9608),
        (277.5, 360.0, 580.5, 624.75, 0.9608),
        (360.0, 442.5, 580.5, 624.75, 0.9608),
        (442.5, 513.0, 580.5, 624.75, 0.9608),
        (513.0, 576.0, 580.5, 624.75, 0.9608),
        (21.75, 576.0, 624.75, 633.0, 0.9608),
        (21.75, 238.5, 675.0, 708.75, 0.9608),
        (238.5, 277.5, 675.0, 708.75, 0.9608),
        (277.5, 360.0, 675.0, 708.75, 0.9608),
        (360.0, 442.5, 675.0, 708.75, 0.9608),
        (442.5, 513.0, 675.0, 708.75, 0.9608),
        (513.0, 576.0, 675.0, 708.75, 0.9608),
        (21.75, 576.0, 708.75, 717.0, 0.9608),
    ],
    words=[
        ('Asset', 25.32, 51.34, 382.62),
        ('Owner', 241.32, 274.07, 382.62),
        ('Value', 280.32, 308.21, 382.62),
        ('of', 310.61, 320.33, 382.62),
        ('Asset', 322.73, 348.74, 382.62),
        ('Income', 362.82, 399.59, 382.62),
        ('Type(s)', 401.99, 438.67, 382.62),
        ('Income', 445.32, 482.09, 382.62),
        ('Income', 515.82, 552.59, 382.62),
        ('Current', 445.32, 478.19, 393.48),
        ('Year', 480.23, 499.66, 393.48),
        ('to', 501.7, 510.0, 393.48),
        ('Preceding', 515.82, 557.43, 393.48),
        ('Filing', 445.32, 469.34, 402.48),
        ('Year', 515.82, 535.26, 402.48),
        ('Adair', 25.2, 47.26, 422.72),
        ('Ranch', 49.43, 74.92, 422.72),
        ('[OL]', 77.1, 95.98, 422.72),
        ('$250,001', 280.2, 318.33, 422.72),
        ('-', 320.5, 323.87, 422.72),
        ('Business', 362.7, 397.73, 422.72),
        ('income', 399.9, 429.07, 422.72),
        ('None', 445.2, 466.62, 422.72),
        ('$50,001', 515.7, 548.8, 422.72),
        ('-', 550.97, 554.34, 422.72),
        ('$500,000', 280.2, 320.48, 433.22),
        ('$100,000', 515.7, 555.1, 433.22),
        ('L\x00\x00\x00\x00\x00\x00\x00:', 25.05, 64.19, 450.82),
        ('Sisters,', 66.26, 93.84, 450.82),
        ('OR,', 95.9, 110.57, 450.82),
        ('US', 112.63, 123.9, 450.82),
        ('D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00:', 25.05, 76.18, 463.57),
        ('Stud', 78.25, 95.82, 463.57),
        ('Farm', 97.88, 118.35, 463.57),
        ('Business', 120.41, 153.69, 463.57),
        ('(Horses)', 155.76, 188.77, 463.57),
        ('Wells', 25.2, 47.37, 489.47),
        ('Fargo', 49.54, 72.59, 489.47),
        ('Accounts', 74.76, 111.31, 489.47),
        ('[BA]', 113.49, 132.16, 489.47),
        ('$1,001', 280.2, 306.89, 489.47),
        ('-', 309.06, 312.43, 489.47),
        ('$15,000', 314.6, 347.71, 489.47),
        ('Interest', 362.7, 394.01, 489.47),
        ('$1', 445.2, 454.55, 489.47),
        ('-', 456.72, 460.09, 489.47),
        ('$200', 462.26, 483.82, 489.47),
        ('$201', 515.7, 535.6, 489.47),
        ('-', 537.77, 541.14, 489.47),
        ('$1,000', 543.31, 571.66, 489.47),
        ('Adair', 25.2, 47.26, 520.22),
        ('Ranch', 49.43, 74.92, 520.22),
        ('$50,001', 280.2, 313.3, 520.22),
        ('-', 315.47, 318.84, 520.22),
        ('Proceeds', 362.7, 398.57, 520.22),
        ('from', 400.74, 420.13, 520.22),
        ('None', 445.2, 466.62, 520.22),
        ('$50,001', 515.7, 548.8, 520.22),
        ('-', 550.97, 554.34, 520.22),
        ('⇒', 77.09, 84.89, 520.25),
        ('$100,000', 280.2, 319.6, 530.72),
        ('horse', 362.7, 384.71, 530.72),
        ('sales', 386.88, 406.12, 530.72),
        ('$100,000', 515.7, 555.1, 530.72),
        ('Horses', 25.2, 53.2, 531.47),
        ('[RP]', 55.37, 73.93, 531.47),
        ('L\x00\x00\x00\x00\x00\x00\x00:', 25.05, 64.19, 549.07),
        ('Sisters,', 66.26, 93.84, 549.07),
        ('OR,', 95.9, 110.57, 549.07),
        ('US', 112.63, 123.9, 549.07),
        ('D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00:', 25.05, 76.18, 561.82),
        ('Horses', 78.25, 104.85, 561.82),
        ('being', 106.91, 127.74, 561.82),
        ('bred', 129.8, 147.14, 561.82),
        ('and', 149.2, 163.47, 561.82),
        ('held', 165.53, 182.0, 561.82),
        ('for', 184.06, 194.95, 561.82),
        ('eventual', 197.01, 229.2, 561.82),
        ('sale', 231.27, 245.85, 561.82),
        ('Deschutes', 25.2, 66.02, 587.72),
        ('County', 68.2, 96.86, 587.72),
        ('Defined', 99.03, 130.52, 587.72),
        ('Contribution', 132.69, 184.2, 587.72),
        ('Plan', 186.37, 204.29, 587.72),
        ('$1,001', 280.2, 306.89, 587.72),
        ('-', 309.06, 312.43, 587.72),
        ('$15,000', 314.6, 347.71, 587.72),
        ('Tax-Deferred', 362.7, 415.97, 587.72),
        ('⇒', 206.46, 214.25, 587.75),
        ('Nationwide', 25.2, 71.34, 598.97),
        ('S&P', 73.51, 90.44, 598.97),
        ('500', 92.61, 108.41, 598.97),
        ('Index', 110.59, 133.47, 598.97),
        ('Fund', 135.64, 156.7, 598.97),
        ('Service', 158.87, 187.5, 598.97),
        ('Class', 189.67, 210.34, 598.97),
        ('(GRMSX)', 25.2, 64.57, 609.47),
        ('[MF]', 66.75, 87.24, 609.47),
        ('Merrill', 25.2, 53.05, 640.22),
        ('Lynch', 55.22, 79.73, 640.22),
        ('Spouse', 81.9, 110.36, 640.22),
        ('IRA', 112.53, 128.39, 640.22),
        ('$1,000,001', 280.2, 325.89, 640.22),
        ('-', 328.06, 331.43, 640.22),
        ('Tax-Deferred', 362.7, 415.97, 640.22),
        ('⇒', 130.56, 138.35, 640.25),
        ('$5,000,000', 280.2, 328.43, 650.72),
        ('Apple', 25.2, 48.44, 651.47),
        ('Inc.', 50.61, 65.95, 651.47),
        ('-', 68.12, 71.49, 651.47),
        ('Common', 73.66, 110.31, 651.47),
        ('Stock', 112.48, 134.4, 651.47),
        ('(AAPL)', 136.57, 166.32, 651.47),
        ('[ST]', 168.5, 185.86, 651.47),
        ('Merrill', 25.2, 53.05, 682.22),
        ('Lynch', 55.22, 79.73, 682.22),
        ('Spouse', 81.9, 110.36, 682.22),
        ('IRA', 112.53, 128.39, 682.22),
        ('$50,001', 280.2, 313.3, 682.22),
        ('-', 315.47, 318.84, 682.22),
        ('Tax-Deferred', 362.7, 415.97, 682.22),
        ('⇒', 130.56, 138.35, 682.25),
        ('$100,000', 280.2, 319.6, 692.72),
        ('AT&T', 25.2, 48.76, 693.47),
        ('Inc.', 50.93, 66.27, 693.47),
        ('(T)', 68.44, 80.76, 693.47),
        ('[ST]', 82.94, 100.3, 693.47),
    ],
    text=ADAIR_COVER_TEXT,
)


@pytest.fixture(scope="module")
def landsman():
    return ha.parse_house_annual([_LANDSMAN_PAGE_1, _LANDSMAN_PAGE_2])


@pytest.fixture(scope="module")
def adair():
    return ha.parse_house_annual([_ADAIR_PAGE_1])


# --------------------------------------------------------------- the glyphs

def test_small_cap_headings_arrive_as_nul_not_whitespace():
    """The bug this normalisation exists for, stated as a test.

    `Schedule A: Assets and "Unearned" Income` extracts with its small-cap
    letters replaced by NUL. `\\s` does not match NUL, so any pattern applied
    before normalising matches nothing and reports an empty schedule.
    """
    heading = COVER_TEXT.splitlines()[-1]
    assert "\x00" in heading, "the fixture lost the real bytes"
    assert not heading.startswith("S "), "NUL is not whitespace"

    normalised = ha.normalise_glyphs(heading)
    assert "\x00" not in normalised
    assert normalised.split()[:2] == ["S", "A:"]


# -------------------------------------------------------------- the brackets

def test_every_official_value_tier_resolves_to_itself():
    for text, (low, high) in ha.VALUE_BRACKETS.items():
        parsed = ha.parse_value_bracket(text)
        assert parsed["value_canonical"], text
        assert (parsed["value_min"], parsed["value_max"]) == (low, high), text


def test_the_top_brackets_are_floors_with_no_ceiling():
    """Summing a zero-width top bracket makes a minimum exceed its maximum."""
    top = ha.parse_value_bracket("Over $50,000,000")
    assert (top["value_min"], top["value_max"]) == (50_000_000, None)

    capped = ha.parse_value_bracket("Spouse/DC Over $1,000,000")
    assert (capped["value_min"], capped["value_max"]) == (1_000_000, None)
    assert capped["spouse_capped"] is True


def test_undetermined_has_no_bounds_and_is_not_zero():
    parsed = ha.parse_value_bracket("Undetermined")
    assert parsed["value_unascertainable"] is True
    assert parsed["value_min"] is None and parsed["value_max"] is None
    assert parsed["no_longer_held"] is False


def test_a_value_of_none_is_a_sale_not_a_zero_value_holding():
    """The one that reads as a $0 holding if you take it at face value.

    A filer picks `None` for an asset disposed of during the year that still
    threw off more than $200. Its year-end value really is nothing -- but it
    is an exited position, and only the flag says so.
    """
    sold = ha.parse_value_bracket("None")
    assert sold["no_longer_held"] is True
    assert (sold["value_min"], sold["value_max"]) == (0, 0)
    assert sold["value_unascertainable"] is False

    held = ha.parse_value_bracket("$1 - $1,000")
    assert held["no_longer_held"] is False, (
        "the smallest real bracket is not a sale")


def test_an_empty_value_cell_claims_nothing():
    parsed = ha.parse_value_bracket("")
    assert parsed["value_min"] is None and parsed["value_max"] is None
    assert parsed["value_canonical"] is False
    assert parsed["no_longer_held"] is False


def test_a_bracket_that_wrapped_over_two_lines_still_reads_as_one_tier():
    """`$100,001 -` on one line and `$250,000` on the next is one bracket."""
    parsed = ha.parse_value_bracket("$100,001 - $250,000")
    assert parsed["value_canonical"] is True
    assert (parsed["value_min"], parsed["value_max"]) == (100_001, 250_000)


def test_every_official_income_tier_resolves_to_itself():
    for text, bounds in ha.INCOME_BRACKETS.items():
        assert ha.parse_income_bracket(text) == bounds, text


def test_other_income_is_a_literal_figure_not_a_bracket():
    """Ted Lieu's CY2025 report, verbatim: an office unit sold for $440,000."""
    assert ha.parse_income_bracket("$131,432.00") == (131_432, 131_432)


def test_income_that_was_never_stated_is_not_income_of_zero():
    assert ha.parse_income_bracket("") == (None, None)
    assert ha.parse_income_bracket("Not Applicable") == (None, None)
    # Tax-deferred earnings are not disclosed at all; zero is a claim the
    # filing did not make.
    assert ha.parse_income_bracket("Tax-Deferred") == (None, None)
    assert ha.parse_income_bracket("None") == (0, 0)


# ----------------------------------------------------------------- the cell

def test_a_nested_holding_splits_into_account_and_asset():
    """Verbatim from 10074384. Roughly four rows in five look like this."""
    cell = ha.split_asset_cell(
        "Fidelity Brokerage ⇒ Fidelity Government Money Market Fund "
        "(SPAXX) [MF]")
    assert cell["parent_account"] == "Fidelity Brokerage"
    assert cell["asset_name"] == "Fidelity Government Money Market Fund"
    assert cell["ticker"] == "SPAXX"
    assert cell["asset_type_code"] == "MF"
    assert cell["asset_type"] == "mutual fund"


def test_the_ticker_is_the_assets_own_not_the_accounts_disambiguator():
    """The House numbers repeated account names `(1)`, `(2)`, `(3)`."""
    cell = ha.split_asset_cell(
        "Rockefeller Capital Management (1) ⇒ Amazon.com, Inc. - Common "
        "Stock (AMZN) [ST]")
    assert cell["parent_account"] == "Rockefeller Capital Management (1)"
    assert cell["ticker"] == "AMZN"
    assert cell["asset_name"] == "Amazon.com, Inc. - Common Stock"


def test_nesting_goes_more_than_one_level_deep():
    """Verbatim from 10071800: an account inside an LLC.

    Split on the first arrow instead of the whole chain and the middle
    account stays welded to the front of the asset's name.
    """
    cell = ha.split_asset_cell(
        "Our Hidden Lake LLC ⇒ UBS Brokerage 2 ⇒ Allstate Corporation "
        "5.10% Dep Shares Pfd Ser H (ALL$B) [ST]")
    assert cell["parent_chain"] == ["Our Hidden Lake LLC", "UBS Brokerage 2"]
    assert cell["parent_account"] == "UBS Brokerage 2"
    assert cell["asset_name"] == (
        "Allstate Corporation 5.10% Dep Shares Pfd Ser H")
    assert cell["ticker"] == "ALL$B"


def test_an_asset_with_no_ticker_gets_none_rather_than_a_guess():
    cell = ha.split_asset_cell("Apartment [RP]")
    assert cell["ticker"] is None
    assert cell["asset_name"] == "Apartment"
    assert cell["asset_type_code"] == "RP"
    assert cell["parent_account"] is None


def test_a_cusip_is_not_a_ticker():
    cell = ha.split_asset_cell("US TREASURY BILL (912797LR7) [GS]")
    assert cell["ticker"] is None


# ------------------------------------------------------------------- scans

def test_a_seven_digit_document_id_is_a_scan():
    """Measured across 1,330 filings; the length discriminates, not the digit."""
    assert ha.is_scanned_doc_id("9116162") is True
    assert ha.is_scanned_doc_id("10074384") is False


def test_a_scan_is_reported_rather_than_downloaded_and_read_as_empty():
    """Refused on the document id alone, before a byte is fetched."""
    with pytest.raises(DisclosureUnavailable) as raised:
        ha.fetch_house_annual("9116162", 2025)
    assert "scan" in str(raised.value)


# ---------------------------------------------------------------- the table

def test_the_columns_are_read_from_the_drawn_header_not_hardcoded(landsman, adair):
    """They differ per document, so a fixed x-offset silently mis-columns."""
    assert landsman["layout"] == "annual"
    assert adair["layout"] == "dual_income"
    landsman_asset = _LANDSMAN_PAGE_1["rects"][0]
    adair_asset = _ADAIR_PAGE_1["rects"][0]
    assert landsman_asset["x1"] != adair_asset["x1"], (
        "the two reports draw the Asset column at different widths, which is "
        "the whole reason the edges are measured")


def test_the_report_header_is_read(landsman):
    assert landsman["member"] == "Greg Landsman"
    assert landsman["report_kind"] == "annual"
    assert landsman["calendar_year"] == 2025
    assert landsman["filed_date"] == "2026-04-20"
    assert landsman["as_of"] == "2025-12-31"
    assert landsman["state_district"] == "OH01"


def test_a_candidate_report_is_dated_to_the_period_it_covered(adair):
    """It covers a stub period, not a calendar year."""
    assert adair["report_kind"] == "candidate"
    assert adair["as_of"] == "2025-11-18"


def test_an_entry_split_by_a_page_break_is_one_holding(landsman):
    """The largest single error source if you take each page at face value.

    `Fidelity - 529 Plan ⇒ MA College Portfolio` is the last entry on page 1
    of 10074384 and its location strip is the first thing printed on page 2.
    Unstitched, the location becomes a row of its own.
    """
    row = next(r for r in landsman["rows"]
               if r["asset_name"] == "MA College Portfolio")
    assert row["spans_page_break"] is True
    assert row["location"] == "MA"
    assert row["parent_account"] == "Fidelity - 529 Plan"
    assert row["owner"] == "dependent_child"
    assert (row["value_min"], row["value_max"]) == (15_001, 50_000)
    assert not any(r["asset_name"] is None for r in landsman["rows"]), (
        "a page-break fragment was left behind as a row with no asset")


def test_an_income_bracket_that_wrapped_beside_the_next_line_is_rejoined(landsman):
    """`$1,001 -` sits at top 180.5 and `$2,500` at 190.97, half a point
    above the asset text it wraps beside. Round the top to a key and the
    bracket is torn in two."""
    row = next(r for r in landsman["rows"] if r["ticker"] == "KR"
               and r["asset_type_code"] == "ST")
    assert (row["income_min"], row["income_max"]) == (1_001, 2_500)
    assert row["income_type"] == "Dividends"


def test_a_value_bracket_that_wrapped_over_two_lines_is_rejoined(landsman):
    row = next(r for r in landsman["rows"]
               if r["asset_name"] == "BR COLLEGE 2032 OPTION - A")
    assert (row["value_min"], row["value_max"]) == (100_001, 250_000)
    assert row["value_canonical"] is True


def test_a_holding_sold_during_the_year_keeps_its_income(landsman):
    """Kroger stock, transferred out, value `None`, dividends of $1,001."""
    row = next(r for r in landsman["rows"] if r["ticker"] == "KR"
               and r["asset_type_code"] == "ST")
    assert row["no_longer_held"] is True
    assert (row["value_min"], row["value_max"]) == (0, 0)
    assert row["income_min"] == 1_001
    assert row["owner"] == "spouse"


def test_a_description_strip_does_not_leak_into_the_columns(landsman):
    """It is drawn across the full table width.

    `Shares were forfeited when spouse terminated her employment with 84.51
    (owned by Kroger) in 2025.` puts `spouse` under Owner and `84.51` under
    Value if the strip is not taken out before the columns are read.
    """
    row = next(r for r in landsman["rows"]
               if r["asset_name"] == "Kroger Company - Restricted Stock")
    assert "forfeited" in row["asset_detail"]
    assert row["owner"] == "spouse"
    assert row["value_text"] == "None"
    assert row["income_type"] == "None"
    # The income *type* is None and the income cell was left blank. Blank is
    # not a reported zero, and this parser does not turn one into the other.
    assert row["income_min"] is None


def test_the_strips_are_recognised_by_their_nul_riddled_labels(landsman):
    row = next(r for r in landsman["rows"]
               if r["asset_name"] == "Kroger Company")
    assert row["asset_detail"] == (
        "Stock transferred to Rockefeller Capital Management (1)")
    assert row["location"] is None


def test_a_parent_account_that_also_files_a_row_is_a_container(adair):
    """Adair Ranch is disclosed on its own and again as the account the
    horses sit in. Counting both doubles it."""
    ranch = next(r for r in adair["rows"] if r["asset_name"] == "Adair Ranch")
    assert ranch["is_container"] is True
    assert ranch["depth"] == 1
    assert (ranch["value_min"], ranch["value_max"]) == (250_001, 500_000)

    horses = next(r for r in adair["rows"] if r["asset_name"] == "Horses")
    assert horses["depth"] == 2
    assert horses["parent_chain"] == ["Adair Ranch"]
    assert horses["parent_account"] == "Adair Ranch"
    assert horses["parent_row"] == ranch["row_number"]

    assert ranch["row_number"] not in {h["row_number"] for h in adair["holdings"]}
    assert horses["row_number"] in {h["row_number"] for h in adair["holdings"]}


def test_a_candidate_report_carries_two_income_columns(adair):
    """No `Tx. > $1,000?`; income for the stub year and the year before."""
    ranch = next(r for r in adair["rows"] if r["asset_name"] == "Adair Ranch")
    assert ranch["income_type"] == "Business income"
    assert (ranch["income_min"], ranch["income_max"]) == (0, 0)
    assert (ranch["income_preceding_min"],
            ranch["income_preceding_max"]) == (50_001, 100_000)
    assert ranch["tx_over_1000"] is None, "the column is not on this layout"

    wells = next(r for r in adair["rows"]
                 if r["asset_name"] == "Wells Fargo Accounts")
    assert (wells["income_min"], wells["income_max"]) == (1, 200)
    assert (wells["income_preceding_min"],
            wells["income_preceding_max"]) == (201, 1_000)


def test_the_transaction_box_is_unread_rather_than_reported_as_no(landsman):
    """It is drawn as a vector glyph, not a character.

    Across 6,996 measured rows the cell yields no words at all, and the box
    is byte-identical whether or not it is ticked. `False` here would assert
    that every member reported no qualifying transaction, on the strength of
    nobody having read the answer.
    """
    assert all(r["tx_over_1000"] is None for r in landsman["rows"])


def test_every_row_in_the_fixtures_lands_on_a_canonical_tier(landsman, adair):
    for report in (landsman, adair):
        for row in report["rows"]:
            assert row["value_canonical"], (row["row_number"], row["value_text"])


def test_the_investment_vehicle_section_defines_the_parent_accounts(landsman):
    vehicles = {v["name"]: v for v in landsman["investment_vehicles"]}
    assert "Fidelity - 529 Plan" in vehicles
    assert vehicles["Fidelity - 529 Plan"]["owner"] == "dependent_child"
    assert vehicles["Rockefeller Capital Management (1)"]["interest_pct"] == 100.0
    assert "IPO" not in " ".join(vehicles), (
        "the list ran past its own section into the certification block")


def test_holdings_are_the_leaves_and_rows_are_everything(landsman):
    assert landsman["has_assets_table"] is True
    assert len(landsman["rows"]) >= len(landsman["holdings"])
    assert all(not h["is_container"] for h in landsman["holdings"])


def test_a_report_with_no_schedule_a_says_so_rather_than_inventing_rows():
    """Twenty-five of the 196 filings measured disclose no assets at all.

    Their Schedule A reads `None disclosed.` and draws no table, so there is
    nothing here to mistake for a member who holds nothing.
    """
    empty = ha.parse_house_annual([_page(rects=[], words=[], text=COVER_TEXT)])
    assert empty["has_assets_table"] is False
    assert empty["rows"] == [] and empty["holdings"] == []
    assert empty["member"] == "Greg Landsman"


# ---------------------------------------------------------------- the wire

@live
@requires_sec
def test_a_live_annual_report_parses_end_to_end():
    parsed = ha.fetch_house_annual("10074384", 2025)
    assert parsed["member"] == "Greg Landsman"
    assert parsed["calendar_year"] == 2025
    assert parsed["as_of"] == "2025-12-31"
    assert len(parsed["rows"]) > 100
    assert all(r["asset_name"] for r in parsed["rows"])
    uncanonical = [r for r in parsed["rows"] if not r["value_canonical"]]
    assert not uncanonical, [r["value_text"] for r in uncanonical]


@live
@requires_sec
def test_the_annual_index_addresses_pdfs_by_coverage_year():
    """A CY2025 annual report is filed in 2026 and still lives under 2025/."""
    filings = ha.list_house_annuals(2025)
    annuals = [f for f in filings if f["filing_type"] == "O"]
    assert len(annuals) > 100
    assert all(f["coverage_year"] == 2025 for f in annuals)
    assert all("/financial-pdfs/2025/" in f["source_url"] for f in annuals)
    assert any(f["is_scan"] for f in annuals), (
        "paper filings exist in every year and have to be reported as such")
