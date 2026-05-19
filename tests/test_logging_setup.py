from trading_copilot.logging_setup import configure_logging, get_logger


def test_configure_logging_returns_logger():
    log = configure_logging()
    assert log is not None
    # Should not raise.
    log.info("phase1.smoke", component="test", ok=True)


def test_get_logger_named():
    log = get_logger("trading_copilot.test")
    log.info("named.logger.works")
